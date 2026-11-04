"""Native PyTorch implementations of the four reshape patterns used by WADNet."""

def rearrange(x, pattern, **axes):
    b = x.shape[0]
    if pattern == 'b c h w -> b (h w) c':
        return x.flatten(2).transpose(1, 2)
    if pattern == 'b (h w) c -> b c h w':
        return x.transpose(1, 2).reshape(b, x.shape[-1], axes['h'], axes['w'])
    if pattern == 'b (head c) h w -> b head c (h w)':
        head = axes['head']
        return x.reshape(b, head, x.shape[1] // head, -1)
    if pattern == 'b head c (h w) -> b (head c) h w':
        return x.reshape(b, x.shape[1] * x.shape[2], axes['h'], axes['w'])
    raise ValueError(f'Unsupported reshape: {pattern}')
