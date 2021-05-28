import pyapr


def read(fpath, apr, parts, t=0, channel_name='t', parts_name='particles', tree_parts=None, read_tree=True):

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(read_tree)

    # Read APR and particles from file
    aprfile.open(fpath, 'READ')
    aprfile.read_apr(apr, t=t, channel_name=channel_name)
    aprfile.read_particles(apr, parts_name, parts, apr_or_tree=True, t=t, channel_name=channel_name)

    if tree_parts is not None and read_tree:
        assert isinstance(tree_parts, (pyapr.ShortParticles, pyapr.FloatParticles))
        aprfile.read_particles(apr, parts_name, tree_parts, apr_or_tree=False, t=t, channel_name=channel_name)

    aprfile.close()


def write(fpath, apr, parts, t=0, channel_name='t', parts_name='particles', write_linear=True, write_tree=True, tree_parts=None):

    if not fpath:
        print('Empty path given. Ignoring call to pyapr.io.write')
        return

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(write_tree)
    aprfile.set_write_linear_flag(write_linear)

    # Write APR and particles to file
    aprfile.open(fpath, 'WRITE')
    aprfile.write_apr(apr, t=t, channel_name=channel_name)
    aprfile.write_particles(parts_name, parts, apr_or_tree=True, t=t, channel_name=channel_name)

    if tree_parts is not None and write_tree:
        assert isinstance(tree_parts, (pyapr.ShortParticles, pyapr.FloatParticles))
        aprfile.write_particles(parts_name, tree_parts, apr_or_tree=False, t=t, channel_name=channel_name)

    aprfile.close()


def write_multichannel(fpath, apr, parts_list, t=0, channel_name='t', channel_names_parts=None):

    if not fpath:
        print('Empty path given. Ignoring call to pyapr.io.write')
        return

    if isinstance(parts_list, (tuple, list)):
        for p in parts_list:
            if not isinstance(p, (pyapr.ShortParticles, pyapr.FloatParticles)):
                raise AssertionError(
                    'argument \'parts_list\' to pyapr.io.write_multichannel must be a \
                    tuple or list of pyapr.XParticles objects'
                )
    else:
        raise AssertionError(
            'argument \'parts_list\' to pyapr.io.write_multichannel must be a tuple or list of pyapr.XParticles objects'
        )

    if channel_names_parts is None:
        channel_names_parts = ['particles' + str(i) for i in range(len(parts_list))]

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath, 'WRITE')
    aprfile.write_apr(apr, t=t, channel_name=channel_name)
    for i in range(len(parts_list)):
        aprfile.write_particles(channel_names_parts[i], parts_list[i], t=t)
    aprfile.close()


def read_multichannel(fpath, apr, parts_list, t=0, channel_name='t', channel_names_parts=None):

    if isinstance(parts_list, (tuple, list)):
        for p in parts_list:
            if not isinstance(p, (pyapr.ShortParticles, pyapr.FloatParticles)):
                raise AssertionError(
                    'argument \'parts_list\' to pyapr.io.read_multichannel must be a \
                    tuple or list of pyapr.XParticles objects')
    else:
        raise AssertionError(
            'argument \'parts_list\' to pyapr.io.read_multichannel must be a tuple or list of pyapr.XParticles objects')

    if channel_names_parts is None:
        channel_names_parts = ['particles' + str(i) for i in range(len(parts_list))]

    # Initialize APRFile for I/O
    aprfile = pyapr.io.APRFile()
    aprfile.set_read_write_tree(True)

    # Write APR and particles to file
    aprfile.open(fpath, 'READ')
    aprfile.read_apr(apr, t=t, channel_name=channel_name)
    for i in range(len(parts_list)):
        aprfile.read_particles(apr, channel_names_parts[i], parts_list[i], t=t)
    aprfile.close()
