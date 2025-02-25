import argparse
import sys
import logging
import yaml
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        if "--config" in args:
            args = FlexibleArgumentParser._pull_args_from_config(args)

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = "--" + key[len("--") :].replace("_", "-")
                    processed_args.append(f"{key}={value}")
                else:
                    processed_args.append("--" + arg[len("--") :].replace("_", "-"))
            else:
                processed_args.append(arg)

        return super().parse_args(processed_args, namespace)

    @staticmethod
    def _pull_args_from_config(args: List[str]) -> List[str]:
        """Method to pull arguments specified in the config file
        into the command-line args variable.

        The arguments in config file will be inserted between
        the argument list.

        example:
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        ```python
        $: vllm {serve,chat,complete} "facebook/opt-12B" \
            --config config.yaml -tp 2
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--config', 'config.yaml',
            '-tp', '2'
        ]
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--port', '12323',
            '--tensor-parallel-size', '4',
            '-tp', '2'
            ]
        ```

        Please note how the config args are inserted after the sub command.
        this way the order of priorities is maintained when these are args
        parsed by super().
        """
        assert args.count("--config") <= 1, "More than one config file specified!"

        index = args.index("--config")
        if index == len(args) - 1:
            raise ValueError(
                "No config file specified! \
                             Please check your command-line arguments."
            )

        file_path = args[index + 1]

        config_args = FlexibleArgumentParser._load_config_file(file_path)

        # 0th index is for {serve,chat,complete}
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        args = [args[0]] + config_args + args[1:index] + args[index + 2 :]

        return args

    @staticmethod
    def _load_config_file(file_path: str) -> List[str]:
        """Loads a yaml file and returns the key value pairs as a
        flattened list with argparse like pattern
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        returns:
            processed_args: list[str] = [
                '--port': '12323',
                '--tensor-parallel-size': '4'
            ]

        """

        extension: str = file_path.split(".")[-1]
        if extension not in ("yaml", "yml"):
            raise ValueError(
                "Config file must be of a yaml/yml type.\
                              %s supplied",
                extension,
            )

        # only expecting a flat dictionary of atomic types
        processed_args: List[str] = []

        config: Dict[str, Union[int, str]] = {}
        try:
            with open(file_path, "r") as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logger.error(
                "Unable to read the config file at %s. \
                Make sure path is correct",
                file_path,
            )
            raise ex

        for key, value in config.items():
            processed_args.append("--" + key)
            processed_args.append(str(value))

        return processed_args
