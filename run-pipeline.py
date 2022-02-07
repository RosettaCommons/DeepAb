#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# :noTabs=true:

# (c) Copyright Rosetta Commons Member Institutions.
# (c) This file is part of the Rosetta software suite and is made available under license.
# (c) The Rosetta software is developed by the contributing members of the Rosetta Commons.
# (c) For more information, see http://www.rosettacommons.org. Questions about this can be
# (c) addressed to University of Washington CoMotion, email: license@uw.edu.

## @file   run-pipeline.py
## @brief  re-implementation of run_pipeline.sh script in Python
## @author Sergey Lyskov

import os, sys, collections, subprocess

from argparse import ArgumentParser

if sys.platform.startswith("linux"): _platform_ = "linux" # can be linux1, linux2, etc
elif sys.platform == "darwin" : _platform_ = "macosx"

_platform_arch_ = f'{_platform_}64' if _platform_ == 'linux' else _platform_

def execute(message, command_line):
    print(message)
    subprocess.check_call(command_line, shell=True)


def download_and_install_archive(name, url, prefix, unpack_to_dir, ignore_directories):
    archive_file_name = url.split('/')[-1]
    dir_name = archive_file_name.split('.')[0]

    signature_path = prefix + f'/.signature-{name}'

    signature = f'v0.1 {name} url={url}'

    if os.path.isfile(signature_path):
        with open(signature_path) as f: on_disk_signature = f.read()
    else: on_disk_signature = None

    if on_disk_signature == signature: print(f'Exising setup for package {name} detected at {prefix!r}... using it...')
    else:
        execute('Downloading {name} from {url!r}...', f'cd {prefix} && wget {url}')

        if archive_file_name.endswith('.zip'):
            extra = f'-d {unpack_to_dir} ' if unpack_to_dir else ''
            extra += f'-j ' if ignore_directories else ''
            execute('Unpacking {archive_file_name}...', f'cd {prefix} && unzip {extra}{archive_file_name}')
        else:
            if unpack_to_dir:
                os.makedirs(f'{prefix}/{unpack_to_dir}')
                extra = f'-C {unpack_to_dir} '
            else: extra = ''

            extra += f'--strip-components=1 ' if ignore_directories else ''
            execute('Unpacking {archive_file_name}...', f'cd {prefix} && tar xf {archive_file_name} {extra}')

        os.remove(prefix + '/' + archive_file_name)

        with open(signature_path, 'w') as f: f.write(signature)




def install_dependencies(prefix):

    if not os.path.isdir(prefix): os.makedirs(prefix)

    PackageInfo = collections.namedtuple('PackageInfo', 'name url unpack_to_dir ignore_directories')

    for p in [
        PackageInfo('ensemble_abresnet', 'https://data.graylab.jhu.edu/ensemble_abresnet_v1.tar.gz', None, None),

    ]: download_and_install_archive(p.name, p.url, prefix, p.unpack_to_dir, p.ignore_directories)


def run_pipeline(input, working_dir, prefix):
    ''' run pipeline for user specified `input`
        note that data dependencies will be installed at `{prefix}/ensemble_abresnet` dir
    '''
    print(f'Running pileline script with input: {input!r}\nprefix: {prefix!r}\nworking-dir: {working_dir!r}')


    not-implemented


def main(args) -> None:
    ''' trRosetta2 pipeline script '''

    #print('Benchmark3 daemon entrance point...')

    parser = ArgumentParser(description=main.__doc__)

    parser.add_argument('--install-dependencies-only', help="Only install dependencies at location specified with `--prefix` option and exit" )

    parser.add_argument('-w', '--working-dir', help="Specify path where script intermediate files and results should be stored")

    parser.add_argument('--prefix', required=True, help='Specify path to to where script dependencies should be installed')

    parser.add_argument('input', help='Path to input `.fa` file')

    options = parser.parse_args()

    install_dependencies(options.prefix)

    if options.install_dependencies_only: return

    run_pipeline(options.input, options.working_dir, options.prefix)


if __name__ == "__main__": main(sys.argv)
