import pytest
import torch
from dna_rnn import DNARNN, create_dna_rnn

class TestDNARNN:
    def test_init(self):
        m = DNARNN(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = DNARNN(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_dna_rnn()
        assert isinstance(m, DNARNN)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
