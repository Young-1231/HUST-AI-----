import torch
from torch import nn

class Encoder(nn.Module):
    """
    encoder基类
    """
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
    

class Decoder(nn.Module):
    """
    decoder基类
    """        
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        
    def init_state(self, encoder_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, *args):
        raise NotImplementedError
    

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder 
    
    def forward(self, enc_x, dec_x, *args):
        encoder_output = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(encoder_output, *args)
        return self.decoder(dec_x, dec_state)
    