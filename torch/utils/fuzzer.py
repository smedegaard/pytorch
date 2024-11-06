import bisect
import copy
import inspect
import itertools
import logging
import random
import string
import sys
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    get_args,
    get_origin,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch


def is_optional_type(type_hint) -> bool:
    origin = get_origin(type_hint)

    if origin is Union:
        args = get_args(type_hint)
        return types.NoneType in args

    return False


class ConfigFuzzer:
    def __init__(
        self,
        config_module,
        test_model_fn: Callable,
        seed: int,
        default: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the config fuzzer.

        Args:
            config_module: The module containing the configs to fuzz
            test_model_fn: Function that runs a test model and returns True if successful
        """
        self.seed = seed
        self.config_module = config_module
        self.test_model_fn = test_model_fn
        self.fields = self.config_module._config
        self.logger = self._setup_logger()
        if default is None:
            self.default = {"force_disable_caches": True}
        else:
            self.default = default

    def __repr__(self):
        return f"ConfigFuzzer(config_module={self.config_module}, test_model_fn={self.test_model_fn}, seed={self.seed}, default={self.default})"

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ConfigFuzzer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _get_type_hint(self, obj, name) -> type:
        """Get type hint for a field, falling back to type(default_value) if not found."""
        try:
            hints = get_type_hints(obj)
            return hints.get(name, type(getattr(obj, name)))
        except Exception:
            return type(getattr(obj, name))

    def _generate_value_for_type(self, type_hint: type) -> Any:
        """Generate a random value for a given type using Hypothesis strategies."""
        if type_hint == bool:
            return random.choice([True, False])
        elif type_hint == int:
            return random.randint(-100, 100)
        elif type_hint == float:
            return random.uniform(-100, 100)
        elif type_hint == str:
            characters = string.ascii_letters + string.digits + string.punctuation
            return "".join(
                random.choice(characters) for _ in range(random.randint(1, 20))
            )
        elif getattr(type_hint, "__origin__", None) == list:
            elem_type = type_hint.__args__[0]
            return [
                self._generate_value_for_type(elem_type)
                for _ in range(random.randint(0, 3))
            ]
        elif getattr(type_hint, "__origin__", None) == dict:
            key_type, value_type = type_hint.__args__
            return {
                self._generate_value_for_type(key_type): self._generate_value_for_type(
                    value_type
                )
                for _ in range(random.randint(0, 3))
            }
        elif is_optional_type(type_hint):
            elem_type = type_hint.__args__[0]
            return random.choice([None, self._generate_value_for_type(elem_type)])
        else:
            raise Exception(f"can't process type hint {type_hint}")

    def _set_config(self, field_name: str, value: Any):
        """Set a config value in the module."""
        setattr(self.config_module, field_name, value)

    def _reset_configs(self):
        """Reset all configs to their default values."""
        for field_name, field_obj in self.fields.items():
            self._set_config(field_name, field_obj.default)

    def fuzz_n_tuple(self, n: int, max_combinations: int = 1000):
        """Test every combination of n configs."""
        self._reset_configs()
        self.logger.info(f"Starting {n}-tuple testing with seed {self.seed}")
        random.seed(self.seed)

        for combo in itertools.combinations(self.fields, n):
            print(combo)
            config = copy.deepcopy(self.default)
            skip = False
            for field_name in combo:
                if field_name in config:
                    skip = True
                field = self.fields[field_name]
                value = self._generate_value_for_type(field.value_type)
                config[field_name] = value
            if skip:
                continue

            # for name, val in config.items():
            #     self._set_config(name, val)

            try:
                # some configs can't be set in the options variable, so we had to set above
                comp = torch.compile(options=config)(self.test_model_fn)
                success = comp()
                if not success:
                    self.logger.error(f"Failure with config combination:")
                    for field, value in values:
                        self.logger.error(f"{field.name} = {value}")
                    return False
            except Exception as e:
                traceback.print_exc()
                breakpoint()
                self.logger.error(f"Exception with config combination:")
                for field, value in config.items():
                    self.logger.error(f"{field} = {value}")
                self.logger.error(f"Exception: {str(e)}")
                return False

            max_combinations -= 1
            if max_combinations <= 0:
                self.logger.info("Reached maximum combinations limit")
                break

        return True

    def fuzz_random_with_bisect(self, num_attempts: int = 100):
        """Randomly test configs and bisect to minimal failing configuration."""
        self.logger.info(f"Starting random testing with bisection and seed {self.seed}")
        random.seed(self.seed)

        for attempt in range(num_attempts):
            self.logger.info(f"Random attempt {attempt + 1}/{num_attempts}")

            # Generate random configs
            test_configs = []
            for field in self.fields:
                if random.random() < 0.3:  # 30% chance to include each config
                    value = self._generate_value_for_type(field.value_type)
                    test_configs.append((field, value))

            # Test the configuration
            self._reset_configs()
            for field, value in test_configs:
                self._set_config(field, value)

            try:
                success = self.test_model_fn()
                if not success:
                    self.logger.info("Found failing configuration, starting bisection")
                    minimal_failing_config = self._bisect_failing_config(test_configs)
                    self.logger.error("Minimal failing configuration:")
                    for field, value in minimal_failing_config:
                        self.logger.error(f"{field.name} = {value}")
                    return False
            except Exception as e:
                self.logger.error(f"Exception during testing: {str(e)}")
                minimal_failing_config = self._bisect_failing_config(test_configs)
                self.logger.error("Minimal failing configuration:")
                for field, value in minimal_failing_config:
                    self.logger.error(f"{field.name} = {value}")
                return False

        self.logger.info("All random tests passed")
        return True

    def _bisect_failing_config(self, failing_configs):
        """Bisect a failing configuration to find minimal set of configs that cause failure."""
        if len(failing_configs) <= 1:
            return failing_configs

        mid = len(failing_configs) // 2
        first_half = failing_configs[:mid]
        second_half = failing_configs[mid:]

        # Test first half
        self._reset_configs()
        for field, value in first_half:
            self._set_config(field, value)

        try:
            if not self.test_model_fn():
                return self._bisect_failing_config(first_half)
        except Exception:
            return self._bisect_failing_config(first_half)

        # Test second half
        self._reset_configs()
        for field, value in second_half:
            self._set_config(field, value)

        try:
            if not self.test_model_fn():
                return self._bisect_failing_config(second_half)
        except Exception:
            return self._bisect_failing_config(second_half)

        # If neither half fails on its own, we need both
        return failing_configs


def create_simple_test_model():
    """Create a simple test model function for demonstration."""

    def test_fn():
        print("in testfn")
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
            )

            x = torch.randn(32, 10)
            model = torch.compile(model)
            y = model(x)
            return True
        except Exception as e:
            print(f"Model test failed: {str(e)}")
            return False

    return test_fn


def main():
    # Example usage
    test_model = create_simple_test_model()
    import torch._inductor.config as cfg

    fuzzer = ConfigFuzzer(cfg, test_model, seed=0)

    # Test every pair of configs
    fuzzer.fuzz_n_tuple(2, max_combinations=100)

    # Test random configs with bisection
    # fuzzer.fuzz_random_with_bisect(num_attempts=50)


if __name__ == "__main__":
    main()
