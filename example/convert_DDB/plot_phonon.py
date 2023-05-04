from abipy.dfpt.ddb import DdbFile


def plot_phonon(fname, qpoints):
    ddb: DdbFile = DdbFile.from_file(fname)
    phbands = ddb.anaget_phmodes_at_qpoints(qpoints)
