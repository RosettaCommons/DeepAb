#!/usr/bin/env python
# -*- coding: utf-8 -*-
# :noTabs=true:

# (c) Copyright Rosetta Commons Member Institutions.
# (c) This file is part of the Rosetta software suite and is made available under license.
# (c) The Rosetta software is developed by the contributing members of the Rosetta Commons.
# (c) For more information, see http://www.rosettacommons.org. Questions about this can be
# (c) addressed to University of Washington CoMotion, email: license@uw.edu.

## @file   dummy.py
## @brief  self-test and debug-aids tests
## @author Sergey Lyskov

import os, os.path, shutil, re, string
import json

import random

import imp
imp.load_source(__name__, '/'.join(__file__.split('/')[:-1]) +  '/__init__.py')  # A bit of Python magic here, what we trying to say is this: from __init__ import *, but init is calculated from file location

_api_version_ = '1.0'


def run_pipeline_test(repository_root, working_dir, platform, config):

    # if os.path.isfile( os.getenv("HOME") + '/.condarc'): raise BenchmarkError(f'~/.condarc file seems to be present, - this _really_ should not have happened, terminating...')
    # else:
    #     if os.path.isfile( os.getenv("HOME") + '/.condarc.template' ) execute('Creating .condarc from template...', 'cp ~/.condarc.template ~/.condarc')

    conda = setup_conda_virtual_environment(working_dir, platform, config)

    execute('Installing PyRosetta package...', f'{conda.activate} && conda install --channel file://{config["mounts"]["release_root"]}/PyRosetta4/conda/release --yes pyrosetta')

    execute('Installing Conda packages...', f'{conda.activate_base} && conda env update --prefix {conda.root} -f {repository_root}/casp14-baker-linux-gpu.yml')

    prefix = calculate_unique_prefix_path(platform, config) + '/trRosetta2'

    execute('Running pipeline script...', f'cd {repository_root} && ./run-pipeline.py --prefix {prefix} --working-dir {working_dir} example/T1078.fa')

    return {_StateKey_ : _S_passed_,  _ResultsKey_ : {},  _LogKey_ : f'' }




def run(test, repository_root, working_dir, platform, config, hpc_driver=None, verbose=False, debug=False):
    if   test == 'pipeline': return run_pipeline_test(repository_root, working_dir, platform, config)

    else: raise BenchmarkError(f'Dummy test script does not support run with test={test!r}!')
