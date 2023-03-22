# DO NOT CHANGE THIS FILE!!!
"""
For the User
-------------

This software  needs some configuration setting to function, including some secrets, like authentication details. \n
We use dynaconf (www.dynaconf.com/) to manage settings and secrets. This tool natively supports a layered approach that
sources settings from files in multiple formats and from environment variables. \n

The package contains some default values for non-secret parameters, encoded into the package source code. \n
The user can partially or totally override these values by providing a custom settings.yaml file. \n
By default the program looks into the user home directory (<home_dir>) for a folder
that has the name of the package (<package_name>) and for a settings.yaml and .secrets.yaml files inside it. \n
\n
If the directory or the necessary files are missing the program stops with an error message: it is necessary to create them! \n
Tha package provides a convenient cli to create them, using defaults

.. code-block:: shell

    <package_name>_configure

By executing this command in the environment where the package is installed, the command will generate the missing files. \n
The settings.yaml file will be populated using the encoded default values \n
If required, .secrets.yaml is also created containing only keys, leaving it to the user to complete it. \n
\n
The program uses the values provided in the settings.yaml and .secrets.yaml files, overriding the encoded defaults. \n

It is possible to override the instruction provided settings.yaml by pointing to another settings file:\n
the values provided in this file override the values in the <home_dir>/<package_name>/.settings.yaml file. \n
This is done by setting the environment variable INCLUDES_FOR_DYNACONF and pointing it to a user-specific setting file
(https://www.dynaconf.com/configuration/)

.. code-block:: shell

    export INCLUDES_FOR_DYNACONF='/path/to/user_specific_settings.yaml'

NOTE for Windows users:
it is possible to set environment variables for each users or globally using the OS

1. Open the Start menu and type "edit ENV" into the search bar
2. Tap "Edit Environment variables for your account"
3. Press "Environment variables" in the "System Variables" section
4. Click "Edit"
5. In the window that opens, hit "Variable value" and input the path you would like to add

the commands to set and to visualize environment variables are, respectively:


.. code-block:: shell

    set ENV_KEY='env_value'
    echo %ENV_KEY%

..

For the Developer
-----------------
The subpackage "configuration" contains a config.py module: do not change this file because this contains the main module interface to the package. \n
Te developer can/should modify the module defaults.py. This module exposes two dictionaries to the config.py module:  \n
1. settings = {"key1": "value1", "key2": "value2"}
2. secrets = { "key1": "<put the correct value here and keep it secret>", "key2": "<put the correct value here and keep it secret>"}

While we populate the settings keys, we leave placeholders for those in secrets. This reflects on the files create by the configuration command:
we do not encode the secrets value in the code!

"""
from pathlib import Path

import yaml
from dynaconf import Dynaconf

from . import defaults

package_name = defaults.package_name


class ConfigurationFileHandler:
    """ class to handle (search, check, create, configuration files)

    Attributes:
    -----------
        user_home_directory: path
            pointer to the user home directory

        config_dir: path
            pointer to the user home directory

        settings_file: path
            pointer to the settings file (settings.yaml)

        secrets_file: path
            pointer to the secret file (.secrets.yaml)

    Methods:
    --------
        check()
            checks the existence of the necessary directory and files

        create()
            create the necessary directory and files

        assess()
            assess if the program can be executed correctly

   """

    user_home_directory = Path.home()
    config_dir = user_home_directory / package_name
    settings_file = config_dir / 'settings.yaml'
    secrets_file = config_dir / '.secrets.yaml'

    def check(self) -> bool:
        """
        check if the necessary configuration files are in the expected directory

               Parameters:
               -----------


               Returns:
               --------

        """
        conf_dir_exists = self.config_dir.exists()
        settings_file_exists = self.settings_file.exists()
        secrets_file_exists = self.secrets_file.exists()

        if not conf_dir_exists:
            print(f"The configuration directory does not exist!: {self.config_dir}")

        if self.config_dir.exists() and not self.config_dir.is_dir():
            raise FileExistsError("a file exists with the name of the configuration directory: please remove it")

        if not settings_file_exists:
            print(f"\nThe settings file does not exist!: {self.settings_file}")

        if not secrets_file_exists:
            print(f"\nThe secrets file does not exist!: {self.secrets_file}")

        return conf_dir_exists and settings_file_exists and secrets_file_exists

    def assess(self) -> bool:
        """
        assess if the program can be executed correctly

               Parameters:
               -----------


               Returns:
               --------

        """

        is_success = self.check()
        if not is_success:
            raise FileNotFoundError("the configuration directory/files are not properly setup")

    def create(self):
        """
        create  the necessary configuration files are in the expected directory

               Parameters:
               -----------


               Returns:
               --------

        """

        if self.config_dir.exists() and not self.config_dir.is_dir():
            raise FileExistsError("a file exists with the name of the configuration directory")

        if not self.config_dir.exists():
            print(f"The configuration directory will be created: {self.config_dir}")
            self.config_dir.mkdir(exist_ok=True)

        for (name, file, content) in zip(('settings', 'secrets'),
                                         (self.settings_file, self.secrets_file),
                                         (defaults.settings, defaults.secrets)):

            if not file.exists():
                write = True
            else:
                with open(file) as stream:
                    content_old = yaml.load(stream, Loader=yaml.Loader)
                message = f'\nThe {name} file was found in {file} \n' \
                          f' current  file content: {content_old}\n' \
                          f' proposed file content: {content}'

                data = _ask(message=message,
                            question='do you what to overwrite it with the new content?',
                            accepted_values=['y', 'n'])
                write = {'y': True, 'n': False}.get(data)
            if write:
                print(f"Writing the {name} file in: {file}")
                with open(file, 'w') as f:
                    yaml.dump(content, f)


def _ask(message: str, question: str, accepted_values: list):
    print(message)
    while True:
        data = input(f" {question} [{'/'.join(accepted_values)}]")
        if data not in accepted_values:
            print(" that is not an appropriate choice.\n")
        else:
            break
    return data


def configure():
    """
    This function it is used to create a cli (command line interface) to configure the parameters after package installation.
    The name of the cli is defined in the project.yaml file of the package and the cli itself is created automatically
    by the package installation procedure.
    Please refer to the package documentation for more details



           Parameters:
           -----------


           Returns:
           --------

    """
    print("Configuring the application")
    cfh = ConfigurationFileHandler()
    cfh.create()


cfh = ConfigurationFileHandler()

# cfh.assess()

settings = Dynaconf(
    settings_files=[cfh.settings_file, cfh.secrets_file],
    envvar_prefix=package_name.capitalize(),
    environments=False
)
