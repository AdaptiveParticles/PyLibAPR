import subprocess
import os
import glob
import shutil

#
#   This script fixes wheels in this folder for windows builds, copying files from the release directory
#   then repacking the wheel and then using delvewheel to fix dependencies.
#

os.chdir('dist')
files = glob.glob('*.whl')
print(files)

for f in files:
    subprocess.check_call(['wheel','unpack',f])

    folders = glob.glob('PyLibAPR_test-*/')

    print('going to loop')
    for folder in folders:
        path = folder + 'Release' + '/*'
        files_2_copy = glob.glob(path)
        for fc in files_2_copy:
            print(fc)
            shutil.copy(fc,folder)
        folder_release = folder + 'Release'
        shutil.rmtree(folder_release)

        subprocess.check_call(['wheel', 'pack', folder])
        shutil.rmtree(folder)
    subprocess.check_call(['delvewheel', 'repair','--ignore-in-wheel',f])

