class Residue():
    """
    Class storing residue identity
    """
    def __init__(self, index: int, identity: str):
        super().__init__()

        assert type(index) == int

        assert type(identity) == str
        assert len(identity) == 1

        self.index = index
        self.identity = identity

    def get_cb_or_ca_atom(self):
        if self.identity == "G":
            return "CA"
        else:
            return "CB"