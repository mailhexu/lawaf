class GPAWWrapper():
    def __init__(self, calc):
        self.calc=calc

    def get_kpoints(self):
        return 

    def get_Ham_and_S(self):
        pass

#from gpaw import restart
#from gpaw.lcao.tools import get_bfi

def test():
    atoms, calc=restart('/home/hexu/projects/jiahui/STO_bulk_LCAO/STO3_gs.gpw', fixdensity=True )
    kpoints=calc.get_ibz_kpoints()
    H, S= get_lcao_hamiltonian()
    desc=calc.setups[0].basis.get_description()


if __name__=="__main__":
    test()
