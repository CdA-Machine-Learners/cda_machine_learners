'''

Configuration alternatively comes from a config file, or is overridden by a
command line argument.


ORDER OF PRIORITIES

1.) CLI arguments
2.) ./config.yml
3.) if not found, then read  ~/config.yml

'''

import argparse
import yaml
import os
import sys
import shutil
import logging
from discord_bot.common import mk_logger
import pkg_resources

log = mk_logger('CONFIG', logging.WARN)

CONFIG_PATHS = [
    'config.yml',
    'config.yaml',
    os.path.expanduser('~/config.yaml'),
    os.path.expanduser('~/config.yml'),
]


def handle_missing_config():
    '''Possibly create a default config, and then exit.'''
    log.error('No config file found!')
    ans = input('Would you like this process to copy the default `config.yml.example` config file into the current directory? (y)es / (n)o')

    if ans.lower() in ['y', 'yes']:
        log.info('''
Copying `config.yml.example` to `config.yml`
Please review it before running the LSP again.
It requires secrets (eg OpenAI Key) so you may prefer to locate it at `~/config.yml`.'''.strip())

        # New path
        config_path = 'config.yml'

        # Example path
        example_config_path = pkg_resources.resource_filename('discord_bot', 'config.yml.example')

        if not os.path.exists(config_path):
            shutil.copyfile(example_config_path, config_path)
    else:
        log.error('''Please manually copy and update the file `config.yml.example`''')
    sys.exit(1)


def load_config(file_paths=CONFIG_PATHS):
    ''' Return first config file that exists. '''
    for file_path in file_paths:
        if os.path.exists(file_path):
            log.info(f'Reading configuration from: {file_path}')
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)

def get_args():
    ''' This first pass will learn generic LSP-related config, and what further
    modules need to be loaded. Those modules will be able to specify their own
    configuration, which will be gathered in a second round of config parsing.
    '''
    # Load config file
    config_yaml = load_config(CONFIG_PATHS)

    if config_yaml is None:
        # Will exit after possibly copying a new config
        handle_missing_config()

    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # LSP-related config
    parser.add_argument('--modules', default=config_yaml.get('modules', None))
    parser.add_argument('--discord_token', default=config_yaml.get('discord_token', None))
    parser.add_argument('--discord_bot_id', default=config_yaml.get('discord_bot_id', None))
    parser.add_argument('--openai_api_key', default=config_yaml.get('openai_api_key', None))

    return parser.parse_args(), config_yaml, parser
