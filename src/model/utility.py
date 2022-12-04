__author__ = 'yawli'
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import numpy.linalg as linalg
import numpy as np
from model import common
import matplotlib.pyplot as plt

# when use_data = True, please pay attention to the following three variables, i.e., count_data, count_conv2d, save_dir.
tol = 1e-2



def singular_value_dist(save_dir, s, n, l):
    if not os.path.exists(os.path.join(os.path.dirname(save_dir), 'svd_dist')):
        os.makedirs(os.path.join(os.path.dirname(save_dir), 'svd_dist'))
    x = list(range(1, s.shape[0] + 1))
    plt.plot(x, s, 'b')
    plt.plot(n, s[n-1], 'ro')
    plt.grid()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Singular value distribuation in Layer {}'.format(l))
    plt.savefig(os.path.join(os.path.dirname(save_dir), 'svd_dist/svd_l{}.png'.format(l)))
    plt.close()

    s_percent = s.cumsum() / s.sum()
    plt.plot(x, 100 * s_percent, 'b')
    plt.plot(n, 100 * s_percent[n-1], 'ro')
    plt.grid()
    plt.xlabel('Index')
    plt.ylabel('Percentage %')
    plt.title('Cumulative energy of singular value in Layer {}'.format(l))
    plt.savefig(os.path.join(os.path.dirname(save_dir), 'svd_dist/psvd_l{}.png'.format(l)))
    plt.close()


def sqrtm_torch_acc(m):

    assert torch.allclose(m, m.t()), "The matrix is not symmetric."
    e, v = m.symeig(eigenvectors=True)
    root = torch.mm(torch.mm(v, e.sqrt().diag()), v.t())
    # embed()
    print('\tMatrix square root: distance {:2.6f}, matrix m norm {:2.6f}'.
          format(torch.dist(m, torch.mm(root, root)).cpu().detach().numpy(), tol * torch.norm(m).cpu().detach().numpy()))
    assert torch.dist(m, torch.mm(root, root)) <= (tol * torch.norm(m)), "The result is not the matrix square root."
    return root


def sqrtm_torch(m):

    assert torch.allclose(m, m.t()), "The matrix is not symmetric."
    u, s, v = m.svd()
    root = torch.mm(torch.mm(u, s.sqrt().diag()), v.t())
    print('\tMatrix square root: distance {:2.6f}, matrix m norm {:2.6f}'.
          format(torch.dist(m, torch.mm(root, root)).cpu().detach().numpy(), tol * torch.norm(m).cpu().detach().numpy()))
    assert torch.dist(m, torch.mm(root, root)) <= (tol * torch.norm(m)), "The result is not the matrix square root."
    return root


def sqrtm_numpy(m):
    assert np.allclose(m , m.T), "The matrix is not symmetric."
    u, s, v = linalg.svd(m)
    root = np.dot(np.dot(u, np.sqrt(np.diag(s))), v)
    print('\tMatrix square root: distance {:2.6f}, matrix m norm {:2.6f}'.
          format(linalg.norm(m - np.dot(root, root)), tol * linalg.norm(m)))
    assert linalg.norm(m - np.dot(root, root)) <= (tol * linalg.norm(m)), "The result is not the matrix square root."
    return root


def gsvd_torch(a, m, w):
    """
    :param a: Matrix to GSVD
    :param m: 1st Constraint, (u.T * m * u) = I
    :param w: 2nd Constraint, (v.T * w * v) = I
    :return: (u ,s, v)
    a = u * diag(s) * v.T  st. (u.T * m * u) = I, and (v.T * w * v) = I
    """

    (aHeight, aWidth) = a.shape
    (mHeight, mWidth) = m.shape
    (wHeight, wWidth) = w.shape

    assert(aHeight == mHeight), "Height not consistent."
    assert(aWidth == wWidth), "Weight not consistent"
    mSqrt = sqrtm_torch(m)
    wSqrt = sqrtm_torch(w)

    mSqrtInv = mSqrt.inverse()
    wSqrtInv = wSqrt.inverse()

    _a = torch.mm(torch.mm(mSqrt, a), wSqrt)

    (_u, _s, _v) = _a.svd()

    u = torch.mm(mSqrtInv, _u)
    v = torch.mm(wSqrtInv, _v)
    s = _s
    print('\tGSVD: distance {:2.6f}, matrix a norm {:2.6f}.'.
          format(torch.dist(a, torch.mm(torch.mm(u, s.diag()), v.t())).cpu().detach().numpy(), tol * torch.norm(a).cpu().detach().numpy()))
    assert torch.dist(a, torch.mm(torch.mm(u, s.diag()), v.t())) <= (tol * torch.norm(a)), "The result is not the GSVD."
    return u, s, v.t()


def gsvd_torch_acc(a, m, w):
    """
    :param a: Matrix to GSVD
    :param m: 1st Constraint, (u.T * m * u) = I
    :param w: 2nd Constraint, (v.T * w * v) = I
    :return: (u ,s, v)
    a = u * diag(s) * v.T  st. (u.T * m * u) = I, and (v.T * w * v) = I
    """

    (aHeight, aWidth) = a.shape
    (mHeight, mWidth) = m.shape
    (wHeight, wWidth) = w.shape

    assert(aHeight == mHeight), "Height not consistent."
    assert(aWidth == wWidth), "Weight not consistent"
    mSqrt = sqrtm_torch_acc(m)
    wSqrt = sqrtm_torch_acc(w)

    mSqrtInv = mSqrt.inverse()
    wSqrtInv = wSqrt.inverse()

    _a = torch.mm(torch.mm(mSqrt, a), wSqrt)

    (_u, _s, _v) = _a.svd()

    u = torch.mm(mSqrtInv, _u)
    v = torch.mm(wSqrtInv, _v)
    s = _s
    print('\tGSVD: distance {:2.6f}, matrix a norm {:2.6f}.'.
          format(torch.dist(a, torch.mm(torch.mm(u, s.diag()), v.t())).cpu().detach().numpy(), tol * torch.norm(a).cpu().detach().numpy()))
    assert torch.dist(a, torch.mm(torch.mm(u, s.diag()), v.t())) <= (tol * torch.norm(a)), "The result is not the GSVD."
    return u, s, v.t()


def gsvd(a, m, w):
    """
    :param a: Matrix to GSVD
    :param m: 1st Constraint, (u.T * m * u) = I
    :param w: 2nd Constraint, (v.T * w * v) = I
    :return: (u ,s, v)
    a = u * diag(s) * v.T  st. (u.T * m * u) = I, and (v.T * w * v) = I
    """
    a = a.detach().numpy().astype('float64')
    m = m.detach().numpy().astype('float64')
    w = w.detach().numpy().astype('float64')
    (aHeight, aWidth) = a.shape
    (mHeight, mWidth) = m.shape
    (wHeight, wWidth) = w.shape

    assert(aHeight == mHeight), "Height not consistent."
    assert(aWidth == wWidth), "Weight not consistent"

    mSqrt = sqrtm_numpy(m)
    wSqrt = sqrtm_numpy(w)
    # embed()

    mSqrtInv = linalg.inv(mSqrt)
    wSqrtInv = linalg.inv(wSqrt)

    _a = np.dot(np.dot(mSqrt, a), wSqrt)

    (_u, _s, _v) = linalg.svd(_a)
    _v = _v.T

    u = np.dot(mSqrtInv, _u)
    v = np.dot(wSqrtInv, _v)
    s = _s
    x = np.zeros((aHeight, aWidth))
    np.fill_diagonal(x, s)
    print('\tGSVD: distance {:2.6f}, matrix a norm {:2.6f}.'.
          format(linalg.norm(a - np.dot(np.dot(u, x), v.T)), (tol * linalg.norm(a))))
    assert linalg.norm(a - np.dot(np.dot(u, x), v.T)) <= (tol * linalg.norm(a)), "The result is not the GSVD."
    return torch.from_numpy(u), torch.from_numpy(s), torch.from_numpy(v.T)


def string_parse(string):
    string_parsed = []
    if string != '':
        for g in string.split('+'):
            if g != '':
                if g.find('*') >= 0:
                    b, mul = g.split('*')
                else:
                    b, mul = g, 1
                string_parsed.extend([int(b)] * int(mul))
    return string_parsed


def param_count(model, ignore_linear):
    """
    :param model:
    :param ignore_linear:
    :return: param_conv, param_linear
    """
    param_linear, param_conv = 0, 0
    if not ignore_linear:
        for layer in find_conv(model, nn.Linear):
            param_linear += layer.in_features * layer.out_features
    for l, layer in enumerate(find_conv(model, nn.Conv2d)):
        param_conv += layer.weight.numel()
        # print(layer.weight.numel(), l)
    return [param_conv, param_linear]

# def flops_linear(model):
#     flops = 0
#     for l, layer in find_conv(model, nn.Linear):
#         flops +=

def mse_post(self, params, layers): #TODO check the memory usage here
    sample, a_sep = self.args.sample, self.args.a_sep
    L = len(layers)
    for l, layer in enumerate(layers):
        self.ckp.write_log('\nLayer {}: MSE step of SVD-MSE method.'.format(l))
        tensor_type = torch.double
        tensor_device = torch.device('cuda')
        ng = self.args.n_GPUs
        bs = self.args.batch_size
        Yp = torch.zeros((bs*sample*(self.n_batch), layer.out_channels), dtype=tensor_type)
        Yb = torch.zeros((bs*sample*(self.n_batch), layer.out_channels), dtype=tensor_type)
        # prepare input and output
        for b in range(self.n_batch):
            for dv in range(ng):
                # read stored feature maps of parent and base model
                feat_p = torch.load(os.path.join(self.save_dir, 'Conv2d_{}'.format(l), 'Batch{}_Device{}.pt'.format(b, dv)), map_location=torch.device('cpu'))
                feat_b = torch.load(os.path.join(self.save_dir, 'DConv2d_{}'.format(l), 'Batch{}_Device{}.pt'.format(b, dv)), map_location=torch.device('cpu'))
                N, C, H, W = feat_p['output'].size()
                # random extraction location
                id = np.random.permutation(W * H)
                # extraction of 3x3 input features and the corresponding output features
                for i in range(sample):
                    x, y = np.unravel_index(id[i], (H, W)) # convert flattened index to coordinates
                    yp = feat_p['output'][:, :, y, x]
                    yb = feat_b['output'][:, :, y, x]
                    Yp[b*bs*sample+dv*bs//ng*sample+i*bs//ng:b*bs*sample+dv*bs//ng*sample+i*bs//ng+bs//ng, :] = yp.to(tensor_type).detach()
                    Yb[b*bs*sample+dv*bs//ng*sample+i*bs//ng:b*bs*sample+dv*bs//ng*sample+i*bs//ng+bs//ng, :] = yb.to(tensor_type).detach()

                # print(torch.sum(feat_p['output'] ** 2), torch.sum(feat_b['output'] ** 2), torch.sum((feat_b['output'] - feat_p['output'])**2))
        # optimization
        A0 = linalg.lstsq(Yb.numpy(), Yp.numpy(), rcond=None)[0]
        A1 = linalg.lstsq(torch.mm(Yb.t(), Yb).numpy() + np.identity(Yb.shape[1]), torch.mm(Yb.t(), Yp).numpy(), rcond=None)[0]
        self.ckp.write_log('A0: {}-{}, A1: {}-{}. A0 max {}, min {}. A1 max {}, min {}.'.
              format(A0.dtype, tuple(A0.shape), A1.dtype, tuple(A1.shape), np.max(A0), np.min(A0), np.max(A1), np.min(A1)))
        self.ckp.write_log('Feature map parent {}, base {}, difference {}'.
              format(torch.norm(Yp).numpy(), torch.norm(Yb).numpy(), torch.norm(Yp - Yb).numpy()))

        self.ckp.write_log('Numpy A0: distance before {:2.4f} and after {:2.4f}'.
              format(linalg.norm(Yp.numpy() - Yb.numpy()), linalg.norm(Yp.numpy() - np.dot(Yb.numpy(), A0))))
        self.ckp.write_log('Numpy A1: distance before {:2.4f} and after {:2.4f}'.
              format(linalg.norm(Yp.numpy() - Yb.numpy()), linalg.norm(Yp.numpy() - np.dot(Yb.numpy(), A1))))
        A0 = torch.from_numpy(A0)
        A1 = torch.from_numpy(A1)
        # At2 = torch.from_numpy(A2)
        self.ckp.write_log('Torch A0: distance before {:2.4f} and after {:2.4f}.'.
              format(torch.sqrt(torch.sum((Yp - Yb)**2)).item(), torch.sqrt(torch.sum((Yp - torch.mm(Yb, A0))**2)).item()))
        self.ckp.write_log('Torch A1: distance before {:2.4f} and after {:2.4f}.'.
              format(torch.sqrt(torch.sum((Yp - Yb)**2)).item(), torch.sqrt(torch.sum((Yp - torch.mm(Yb, A1))**2)).item()))

        A1 = A1.to(torch.float)
        if a_sep: # whether to separate matrix as a 1x1 conv
            params[l]['projection2'] = A1.t().view(A1.shape[0], -1, 1, 1)
        else:
            if 'projection' in params[l]:
                params[l]['projection'] = torch.mm(A1.t(), params[l]['projection'].squeeze()).view(params[l]['projection'].shape[0], -1, 1, 1)
                if params[l]['bias'] is not None:
                    params[l]['bias'] = torch.mm(A1.t(), params[l]['bias'].unsqueeze(dim=1)).squeeze()
            else:
                raise NotImplementedError('This mode is not needed.')
    return params

def decompose_gsvd(self, layers):
    """
    :param layers: the conv layers in the parent model
    :param constrain: the gsvd matrix constraint. 'params' means the filter in the parent model is used to
    constrain gsvd decomposition. 'features' means the matrix (X.T * X)^-1 * X.T *Y from the feature map is used
    as the contraint.
    :return: U, S, V - the list of u, s, v of gsvd of all of the layers.
             meta_data - the meta data, i.e., input / output channel size, kernel size, whether there is bias or
             not.
             num_params - the total number of params in the convs layers. Used to compute the actual compression
             ratio
    """
    tensor_type = torch.double
    tensor_device = torch.device('cuda:0')
    ng = self.args.n_GPUs
    bs = self.args.batch_size
    constrain, sample, include_bias = self.args.gsvd_constrain, self.args.sample, self.args.include_bias
    decom_matrix, meta_data = [], []
    L = len(layers)
    for l, layer in enumerate(layers):
        self.ckp.write_log('Decompose Layer {} using GSVD'.format(l))

        # print('1 memory allocated {}'.format(torch.cuda.memory_allocated()/1024.0**3))
        X = torch.zeros((bs*sample*(self.n_batch), layer.in_channels*layer.kernel_size[0]**2), dtype=tensor_type, device=tensor_device)
        Y = torch.zeros((bs*sample*(self.n_batch), layer.out_channels), dtype=tensor_type, device=tensor_device)
        if self.args.n_GPUs > 1:
            X1 = torch.zeros((bs*sample*(self.n_batch), layer.in_channels*layer.kernel_size[0]**2), dtype=tensor_type, device=torch.device('cuda:1'))
            Y1 = torch.zeros((bs*sample*(self.n_batch), layer.out_channels), dtype=tensor_type, device=torch.device('cuda:1'))
        for b in range(self.n_batch):
            for dv in range(ng):
                feat_in, feat_out = [v for v in torch.load(os.path.join(self.save_dir, 'Conv2d_{}'.format(l),
                                     'Batch{}_Device{}.pt'.format(b, dv)), map_location=torch.device('cpu')).values()]
                N, C, H, W = feat_out.size()
                # random feature extraction location
                xy = np.random.permutation(W * H)
                # padding the input, pad size decided by the kernel size.
                p = layer.padding[0]
                stride = layer.stride[0]
                feat_in = F.pad(feat_in, (p, p, p, p), 'constant', 0)
                # extraction of 3x3 input features and the corresponding output features
                for i in range(sample):
                    # convert flattened index to coordinates
                    x, y = np.unravel_index(xy[i], (H, W))
                    xi, yi = stride * x + p, stride * y + p
                    xm = feat_in[:, :, yi-p:yi+p+1, xi-p:xi+p+1].reshape(N, -1)
                    ym = feat_out[:, :, y, x]
                    # embed()
                    # print(b*bs*sample+dv*bs//ng*sample+i*bs//ng, b*bs*sample+dv*bs//ng*sample+i*bs//ng+bs//ng)
                    X[b*bs*sample+dv*bs//ng*sample+i*bs//ng:b*bs*sample+dv*bs//ng*sample+i*bs//ng+bs//ng, :] \
                        = xm.to(tensor_type).to(tensor_device).detach()
                    Y[b*bs*sample+dv*bs//ng*sample+i*bs//ng:b*bs*sample+dv*bs//ng*sample+i*bs//ng+bs//ng, :] \
                        = ym.to(tensor_type).to(tensor_device).detach()

        if include_bias and layer.bias is not None: # bias is also considered
            X = F.pad(X, (0, 1), 'constant', 1)

        if layer.bias is not None:
            self.ckp.write_log('\tweight max {:2.4f}, min {:2.4f}; bias max {}, min {}.'.
                               format(layer.weight.data.max(), layer.weight.data.min(), layer.bias.data.max(), layer.bias.data.min()))
        else:
            self.ckp.write_log('\tweight max {:2.4f}, min {:2.4f}.'.format(layer.weight.data.max(), layer.weight.data.min()))

        covar = torch.mm(X.t(), X) # covariance X.T * X
        ccovar = torch.mm(X.t(), Y) # cross covariance X.T * Y
        self.ckp.write_log('\tX shape {}, Y shape {},  covariance shape {}, cross-covariance shape {}'
              .format(tuple(X.shape), tuple(Y.shape), tuple(covar.shape), tuple(ccovar.shape)))
        del X, Y
        torch.cuda.empty_cache()

        # determine the constraint matrix according to the constraint option.
        c_out, c_in, k, _ = layer.weight.data.size()
        is_inverse = torch.matrix_rank(covar).cpu().detach().numpy() == covar.shape[0]
        self.ckp.write_log('\tMatrix non-singular {}'.format(is_inverse))
        if constrain == 'params':
            weight = layer.weight.data
            weight = weight.view(c_out, -1)
            if include_bias and layer.bias is not None:
                bias = layer.bias.data
                A = torch.cat((weight, bias.unsqueeze(dim=1)), dim=1).t().to(tensor_type).to(tensor_device)
            else:
                A = weight.t().to(tensor_type).to(tensor_device)
        elif constrain == 'features':
            if is_inverse:
                A = torch.mm(covar.inverse(), ccovar)
            else:
                A = torch.mm(torch.pinverse(covar), ccovar)
        else:
            raise NotImplementedError('GSVD target {} is not defined.'.format(constrain))
        gsvd_fun = gsvd_torch_acc if is_inverse else gsvd_torch
        u, s, v = gsvd_fun(A, covar, torch.eye(A.size(1), dtype=tensor_type, device=tensor_device))
        self.ckp.write_log('\tu {}, s {}, v {}.'.format(tuple(u.shape), tuple(s.diag().shape), tuple(v.shape)))
        decom_matrix.append([k.to(torch.device('cpu')) for k in [u, s, v]])
        meta_data.append((c_out, c_in, k, layer.bias))
        del covar, ccovar, A, u, s, v # X and Y are very large matrices. So delete them immediately when they are not useful.
        torch.cuda.empty_cache()

    return decom_matrix, meta_data

# def decompose_svd(self, layers, basis_size):
#
#     basis_size = string_parse(basis_size)
#     decom_matrix, meta_data = [], []
#     num_params = 0
#     for i, layer in enumerate(layers):
#         self.ckp.write_log('Decompose Layer {} using SVD'.format(i))
#         weight = layer.weight.data
#         c_out, c_in, k, _ = weight.size()
#         num_params += c_out * c_in * k ** 2
#
#         bs = c_in if len(basis_size) == 0 else basis_size[i]
#         group = c_in // bs
#         weight = weight.view(c_out * group, -1)
#         if layer.bias is None:
#             params = weight.t()
#         else:
#             bias = layer.bias.data
#             # params = torch.cat((weight, bias.unsqueeze(dim=1)), dim=1).t()
#             params = torch.cat((weight, bias.repeat(group, 1).t().reshape(-1, 1) / group), dim=1).t()
#         decom_matrix.append(list(params.svd()))
#         meta_data.append((c_out, c_in, k, layer.bias))
#
#     return decom_matrix, meta_data, num_params

def decompose_svd(self, layers):
    include_bias = self.args.include_bias
    decom_matrix, meta_data = [], []
    for i, layer in enumerate(layers):
        self.ckp.write_log('Decompose Layer {} using SVD'.format(i))
        weight = layer.weight.data
        c_out, c_in, k, _ = weight.size()
        weight = weight.view(c_out, -1)
        if include_bias and layer.bias is not None:
            bias = layer.bias.data
            params = torch.cat((weight, bias.unsqueeze(dim=1)), dim=1).t()
        else:
            params = weight.t()
        usv = linalg.svd(params.cpu().numpy().astype('float64')) # use numpy arrray
        u, s, v = [torch.from_numpy(i) for i in usv]
        decom_matrix.append([u, s, v])
        meta_data.append((c_out, c_in, k, layer.bias))

        if layer.bias is not None:
            #print(layer.bias)
            self.ckp.write_log('\tweight max {:2.4f}, min {:2.4f}; bias max {}, min {}.'.
                               format(weight.max(), weight.min(), layer.bias.data.max(), layer.bias.data.min()))
        else:
            self.ckp.write_log('\tweight max {:2.4f}, min {:2.4f}.'.format(weight.max(), weight.min()))
        self.ckp.write_log('\tu {}, s {}, v {}.'.format(tuple(u.shape), tuple(s.diag().shape), tuple(v.shape)))
    return decom_matrix, meta_data

def compression(self, decom_matrix, meta_data, parent_layers, base_layers):
    """
    :param parent_model: parent mdel
    :param U: list of u of all layers
    :param S: list of s of all layers
    :param V: list of v of all layers
    :param meta_data: meta data of all conv layers
    :param param_conv: number of params in conv layers
    :param ratio: compression ratio
    :param comp_method: compression type,
    'fixed-rank' - the same compression ratio for all layers,
    'adp-simple' - adaptive for each layer but comp ratio is fixed only in terms of rank reduction,
    'adp-tight' - adaptive and the compression ratio is fixed in terms of number of parameters,
    'manual' - manually determine the rank.
    :param n_basis: used for 'manual' mode
    :param basis_size: used for 'manual' mode
    :return:
    """
    # used to calculate the compression ratio
    comp_method = self.args.comp_method

    if comp_method == 'fixed-rank':
        params = self.reduction_fixed_rank(decom_matrix, meta_data, parent_layers)
    elif comp_method == 'adp-tight':
        params = self.reduction_adp_tight(decom_matrix, meta_data, parent_layers, base_layers)
    elif comp_method == 'adp-simple':
        params = self.reduction_adp_simple(decom_matrix, meta_data, parent_layers)
    elif comp_method == 'manual':
        params = self.reduction_manual(decom_matrix, meta_data, parent_layers)
    else:
        raise NotImplementedError('Network compression type {} is not implemented.'.format(comp_method))
    return params

def conv_full_params(self, decom_matrix, meta_data):
    params = []
    include_bias = self.args.include_bias
    for l, (ml, md) in enumerate(zip(decom_matrix, meta_data)):
        u, s, v = ml[0], ml[1], ml[2]
        param = torch.mm(u[:, :s.shape[0]], s[:].diag()).t().to(torch.float)
        projection = v.t()[:, :s.shape[0]].view(md[0], -1, 1, 1).to(torch.float)
        if not include_bias:
            weight = param.view(-1, md[1], md[2], md[2])
            bias = None if md[3] is None else md[3].cpu()
            param_dict = {'weight': weight, 'projection': projection, 'bias': bias}
        else:
            if md[3] is None:
                weight = param.view(-1, md[1], md[2], md[2])
                bias = None
            else:
                weight = param[:, :-1].view(-1, md[1], md[2], md[2])
                bias = param[:, -1]
            param_dict = {'weight': weight, 'projection': projection, 'bias': bias}
        params.append(param_dict)
    return params

def reduction(self, l, n_filter, matrixl, md):
    conv_single, include_bias = self.args.conv_single, self.args.include_bias
    u, s, v = matrixl[0], matrixl[1], matrixl[2]
    s_ = s.detach().numpy()
    self.ckp.write_log('Reduction: layer {:2}, n_filter={:4}, kernel_size={:1}. {:2.2f}% energy remained. '
        'Singular value distance bound {:2.4f} = {:2.4f} - {:2.4f}, max/bound ratio {:2.4f}. '
        'Max {:2.4f}, mean {:2.4f}, mean start {:2.4f}, min {:2.4f}. '
        .format(l, n_filter, md[2], sum(s_[:n_filter]) / sum(s_),
              s_[n_filter-2] - s_[n_filter-1], s_[n_filter-2], s_[n_filter-1], s_.max() / s_[n_filter-1],
              s_.max(), s_.mean(), s_[:n_filter-1].mean(), s_.min()))

    singular_value_dist(self.save_dir, s_, n_filter, l)
    if not conv_single:

        # param = torch.mm(u[:, :s.shape[0]], s[:s.shape[0]].diag()).t().to(torch.float)
        # projection = v.t()[:, :s.shape[0]].view(md[0], -1, 1, 1).to(torch.float)

        param = torch.mm(u[:, :n_filter], s[:n_filter].diag()).t().to(torch.float)
        projection = v.t()[:, :n_filter].view(md[0], n_filter, 1, 1).to(torch.float)
        if not include_bias:
            weight = param.view(-1, md[1], md[2], md[2])
            bias = None if md[3] is None else md[3].cpu()
            return {'weight': weight, 'projection': projection, 'bias': bias}
        else:
            if md[3] is None:
                weight = param.view(-1, md[1], md[2], md[2])
                bias = None
            else:
                weight = param[:, :-1].view(-1, md[1], md[2], md[2])
                bias = param[:, -1]
            return {'weight': weight, 'projection': projection, 'bias': bias}
        # weight = self.layers[l].weight
        # bias = self.layers[l].bias
        # return {'weight': weight, 'bias': bias}

    else:
        # param = torch.mm(torch.mm(u[:, :s.shape[0]], s[:s.shape[0]].diag()), v[:s.shape[0], :]).t().to(torch.float)
        param = torch.mm(torch.mm(u[:, :n_filter], s[:n_filter].diag()), v[:n_filter, :]).t().to(torch.float)
        if not include_bias:
            weight = param.view(md[0], md[1], md[2], md[2])
            bias = None if md[3] is None else md[3].cpu()
            return {'weight': weight, 'bias': bias}
        else:
            if md[3] is None:
                weight = param.view(md[0], md[1], md[2], md[2])
                bias = None
            else:
                weight = param[:, :-1].view(md[0], md[1], md[2], md[2])
                bias = param[:, -1]
            return {'weight': weight, 'bias': bias}

def find_conv(model, criterion, conv_kernel=0, in_channels=-1):

    layers = []
    for layer in model.modules():
        if isinstance(layer, criterion):
            if hasattr(layer, 'in_channels'):
                if model.__class__.__name__ == 'VGG_ICCV':
                    if layer.in_channels >= 128:
                        layers.append(layer)
                else:
                    if criterion == nn.Conv2d:
                        # for DenseNet-40
                        if conv_kernel == 0:
                            layers.append(layer)
                        elif conv_kernel == layer.kernel_size[0] and layer.in_channels > in_channels:
                        # only keep the convs in DenseBlock
                            layers.append(layer)
                    else:
                        layers.append(layer)
            else:
                layers.append(layer)

    return layers

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def find_index(s, value, rmin=3, rmax_p=0.8):
    n_filter = max(torch.sum(s >= value).numpy(), rmin)
    n_filter = min(n_filter, int(rmax_p * s.shape[0])) # The compression ratio must be smaller than 0.8
    return n_filter

def search_config(gamma, S, info, config_e):
    config = []
    S_sorted, _ = torch.cat(S).sort(descending=True)
    index = int(round(S_sorted.shape[0] * gamma))
    value = S_sorted[index-1] # why index-1, The total number of selected values is index. The value at index is included.
    for l, (s, i) in enumerate(zip(S, info)):
        if config_e is None:
            config.append(find_index(s, value))
        else:
            config.append(find_index(s, value, rmin=config_e[l]))
    return config, value, index

def search_config_energy(E_target, S):
    config_e = []
    for l, s in enumerate(S):
        s_percent = s.numpy().cumsum() / s.numpy().sum()
        idx, val = find_nearest(s_percent, E_target)
        # print('{:2.4f} % of the energy appears in the former {} singular values.'.format(val, idx))
        config_e.append(idx)
    return config_e

def cal_complexity(config, info, num_params):
    num_params_c = 0
    flops_c = 0
    for l, (n_keep, i) in enumerate(zip(config, info)):
        # assert(n_keep * md[1] * md[2] ** 2 + n_keep * md[0] <= md[0] * md[1] * md[2] ** 2),\
        #     'Warning in decomposing layer {}: There are more parameters in the decomposed layer than in the original layer.'.format(l)
        num_params_c += n_keep * i[1] * i[2] ** 2 + n_keep * i[0]
        flops_conv1 = n_keep * i[1] * i[2] ** 2 # kernel_size = 3
        flops_conv2 = i[0] * n_keep * 1 ** 2 # kernel_size = 1
        flops_c += np.prod(i[3]) * 1 * (flops_conv1 + flops_conv2)
    num_params_c += num_params[1] #TODO check all of the num_params
    flops_c += num_params[1] # For nn.Linear, the number of parameters and flops are the same if batch_size = 1
    return flops_c, num_params_c

def metric_based_searching(self, S, info, config_e=None):
    ratio = self.args.ratio
    comp_target = self.args.comp_target
    num_params = self.num_params
    flops = int(self.parent_flops)

    gamma = ratio
    nu_old = 0
    step = 0.1
    i = 1
    Sn = torch.cat(S).shape[0]

    while step > 0.0001 and abs(ratio-nu_old) > 0.001:
        config, value, index = search_config(gamma, S, info, config_e)
        flops_c, num_params_c = cal_complexity(config, info, num_params)

        nu_new = num_params_c / sum(num_params) if comp_target == 'params' else flops_c / flops
        flag = False if i == 1 else (nu_old >= ratio) == (nu_new < ratio)
        if flag: step /= 2
        gamma = gamma - step if nu_new >= ratio else gamma + step
        nu_old = nu_new
        string = 'Iter {}: gamma={:2.4f}, flag={:1}, step={:1.4f}, ratio={:2.4f}. '.format(i, gamma, flag, step, nu_new) \
                 + 'Optimization Target: {}, #params={}/{}, #flops={}/{}. '.format(comp_target, num_params_c, sum(num_params), flops_c, flops) \
                 + 'Singular values: {}/{}.'.format(Sn, index)
        self.ckp.write_log(string)
        i += 1
        if gamma >= 1 or gamma <= 0:
            break
    if gamma >= 1 or gamma <= 0:
        return None, None, None
    else:
        return config, num_params_c, flops_c

def energy_constrained_searching(self, S, info, E_start):
    ratio = self.args.ratio
    comp_target = self.args.comp_target
    num_params = self.num_params
    flops = int(self.parent_flops)

    stop = False
    stepE = 0.1
    tightness = 0.05
    E_target = E_start
    i_ernegy = 1
    enu_old = 0
    while not stop:
        config_e = search_config_energy(E_target, S)
        flops_c, num_params_c = cal_complexity(config_e, info, num_params)

        enu_new = num_params_c / sum(num_params) if comp_target == 'params' else flops_c / flops
        flagE = False if i_ernegy == 1 or stepE <= tightness else (enu_old >= ratio) == (enu_new < ratio)
        if flagE: stepE = stepE / 2
        E_target = E_target - stepE if (enu_new > ratio and stepE > tightness) or stepE <= tightness else E_target + stepE
        enu_old = enu_new
        self.ckp.write_log('Energy constrained searching, iter{}, energy {:2.4f}, flagE {:1}, enu_new {:2.4f}'
                           .format(i_ernegy, E_target, flagE, enu_new))

        if stepE <= tightness:
            config, num_params_c, flops_c= self.metric_based_searching(S, info, config_e)
            if config is None:
                stop = False
            else:
                stop = True
        else:
            stop = False
        i_ernegy += 1
    return config, num_params_c, flops_c

def reduction_adp_tight(self, decom_matrix, meta_data, parent_layers, base_layers):
    """
    parameters in the this code
    gamma: rank shrinkage ratio
    ratio: network compression ratio
    step: searching step for rank shrinkage ratio, using binary search
    i: number of iterations

    """
    normalize = True
    searching_method = self.args.searching_method # choices: energy and metric
    if self.args.comp_rule.find('f-norm') >= 0: # also need to decide whether to normalize S
        S = [base_layers[i].__feature_map_norm__ / (base_layers[i].__feature_map_norm__.max() if normalize else 1) for i in range(len(meta_data))]
    else:
        S = [decom_matrix[i][1] / (decom_matrix[i][1].max() if normalize else 1) for i in range(len(meta_data))]

    def _layer_info(parent_layers, meta_data):
        info = []
        for l in range(len(meta_data)):
            info_per = list(meta_data[l][:3])
            info_per.append(parent_layers[l].__output_dims__)
            print(info_per)
            info.append(info_per)
        return info

    info = _layer_info(parent_layers, meta_data)
    print(info)
    # embed()
    self.ckp.write_log('Searching method {} for the possible configurations'.format(searching_method))
    if searching_method == 'metric':
        config, num_params_c, flops_c = self.metric_based_searching(S, info)
    else:
        E_start = 0.8
        config, num_params_c, flops_c = self.energy_constrained_searching(S, info, E_start)

    params = []
    for l, (ml, md, n_keep) in enumerate(zip(decom_matrix, meta_data, config)):
        params.append(self.reduction(l, n_keep, ml, md))

    self.ckp.write_log('Target:{}, #params={}/{}, #flops={}/{}, '
                       'flops reduction ratio {:2.4f}, network compression ratio {:2.4f}, rank shrinkage ratio {:2.4f}.'
                       .format(self.args.comp_target, num_params_c, sum(self.num_params), flops_c, self.parent_flops,
                        flops_c/self.parent_flops, num_params_c/sum(self.num_params), sum(config) / torch.cat(S).shape[0]))
    return params

def reduction_adp_simple(self, decom_matrix, meta_data, parent_layers):
    ratio = self.args.ratio
    num_params = self.num_params
    S = [decom_matrix[i][1] for i in range(len(decom_matrix))]
    S_total, _ = torch.cat(S).sort(descending=True)
    index = int(S_total.shape[0] * ratio)
    value = S_total[index]

    params = []
    num_params_c = num_params[1]
    rank_sum = 0
    for l, (ml, md) in enumerate(zip(decom_matrix, meta_data)):
        n_filter = max(torch.sum(ml[1] >= value), 3)
        num_params_c += n_filter * md[1] * md[2] ** 2 + n_filter * md[0]
        rank_sum += n_filter
        params.append(self.reduction(l, n_filter, ml, md))

    self.ckp.write_log('original parameters = {}, current parameters = {}, network compression ratio {:2.4f}, rank shrinkage ratio {:2.4f}'
                       .format(sum(num_params), num_params_c, num_params_c / sum(num_params), rank_sum / S_total.shape[0]))
    return params

def reduction_fixed_rank(self, decom_matrix, meta_data, parent_layers):
    ratio = self.args.ratio
    num_params = self.num_params
    flops = int(self.parent_flops)
    params = []
    num_params_c = num_params[1]
    rank_sum = 0
    rank_total = 0
    flops_c = 0
    for l, (ml, md) in enumerate(zip(decom_matrix, meta_data)):
        n_filter = int(md[0] * ratio)
        num_params_c += n_filter * md[1] * md[2] ** 2 + n_filter * md[0]
        rank_sum += n_filter
        rank_total += md[0]
        params.append(self.reduction(l, n_filter, ml, md))
        flops_conv1 = n_filter * md[1] * md[2] ** 2 # kernel_size = 3
        flops_conv2 = md[0] * n_filter * 1 ** 2 # kernel_size = 1
        flops_c += np.prod(parent_layers[l].__output_dims__) * 1 * (flops_conv1 + flops_conv2)
    flops_c += num_params[1]
    self.ckp.write_log('Target:, #params={}/{}, #flops={}/{}, flops reduction ratio {:2.4f}, network compression ratio {:2.4f}, rank shrinkage ratio {:2.4f}.'
                       .format(num_params_c, sum(num_params), flops_c, flops, flops_c/flops, num_params_c/sum(num_params), rank_sum / rank_total))
    # self.ckp.write_log('original parameters = {}, current parameters = {}, network compression ratio {:2.4f}, rank shrinkage ratio {:2.4f}'
    #                    .format(sum(num_params), num_params_c, num_params_c / sum(num_params), rank_sum / rank_total))
    return params

def reduction_manual(self, decom_matrix, meta_data, parent_layers):
    n_basis, basis_size, conv_single = self.args.n_basis_str, self.args.basis_size_str, self.args.conv_single
    include_bias = self.args.include_bias
    num_params = self.num_params
    # U = [decom_matrix[i][0] for i in range(len(decom_matrix))]
    # S = [decom_matrix[i][1] for i in range(len(decom_matrix))]
    # V = [decom_matrix[i][2] for i in range(len(decom_matrix))]
    params = []
    num_params_c = num_params[1]
    n_basis = string_parse(n_basis)
    # embed()
    basis_size = string_parse(basis_size)
    for l, ((u, s, v), md) in enumerate(zip(decom_matrix, meta_data)):
        s_ = s.detach().numpy()
        n_filter = md[0] if len(n_basis) == 0 else n_basis[l]
        bs = md[1] if len(basis_size) ==0 else basis_size[l]
        num_params_c += n_filter * md[1] * md[2] ** 2 + n_filter * md[0]
        # params.append(self.reduction(l, n_filter, ml, md, conv_single))
        self.ckp.write_log('Reduction: layer {:2}, n_filter={:4}, kernel_size={:1}. {:2.2f}% energy remained. '
        'Singular value distance bound {:2.4f} = {:2.4f} - {:2.4f}, max/bound ratio {:2.4f}.'
        'Max {:2.4f}, mean {:2.4f}, mean start {:2.4f}, min {:2.4f}'
        .format(l, n_filter, md[2], sum(s_[:n_filter]) / sum(s_),
              s_[n_filter-2] - s_[n_filter-1], s_[n_filter-2], s_[n_filter-1], s_.max() / s_[n_filter-1],
              s_.max(), s_.mean(), s_[:n_filter-1].mean(), s_.min()))

        singular_value_dist(self.save_dir, s_, n_filter, l)
        param = torch.mm(u[:, :n_filter], s[:n_filter].diag()).t().to(torch.float)
        projection = v.t()[:, :n_filter].reshape(-1, n_filter * md[1] // bs, 1, 1).to(torch.float)
        if not include_bias:
            weight = param.view(-1, md[1], md[2], md[2])
            bias = None if md[3] is None else md[3].cpu().to(torch.float)
        else:
            if md[3] is None:
                weight = param.view(n_filter, md[1], md[2], md[2]) # 128 * 128 * 3 * 3
                bias = None
            else:
                weight = param[:, :-1].view(n_filter, -1, md[2], md[2])
                bias = param[:, -1]
        params.append({'weight': weight, 'projection': projection, 'bias': bias})
    self.ckp.write_log('original parameters = {}, current parameters = {}, network compression ratio {:2.4f}'.
                       format(sum(num_params), num_params_c, num_params_c / sum(num_params)))
    return params

def dconv_init(self, params):
    layers = find_conv(self, common.DConv2d)
    for l, layer in enumerate(layers):
        # weight
        self.register_parameter('weight_{}'.format(l), nn.Parameter(params[l]['weight']))
        # bias
        if params[l]['bias'] is not None:
            self.register_parameter('bias_{}'.format(l), nn.Parameter(params[l]['bias']))
        else:
            self.register_parameter('bias_{}'.format(l), params[l]['bias'])
        # projection
        if 'projection' in params[l]:
            self.register_parameter('projection_{}'.format(l), nn.Parameter(params[l]['projection']))
        # projection2
        if 'projection2' in params[l]:
            self.register_parameter('projection2_{}'.format(l), nn.Parameter(params[l]['projection2']))
        param = {'weight': getattr(self, 'weight_{}'.format(l)),
                 'bias': getattr(self, 'bias_{}'.format(l))}
        if hasattr(self, 'projection_{}'.format(l)):
            param['projection'] = getattr(self, 'projection_{}'.format(l))
        if hasattr(self, 'projection2_{}'.format(l)):
            param['projection2'] = getattr(self, 'projection2_{}'.format(l))
        layer.set_params(param)
        # print(list(param.keys()), layer)

def feat_norm_cal(layers, n_batch, n_gpus, save_dir=''):
    for l, layer in enumerate(layers):
        feature_map_norm = []
        for b in range(n_batch):
            for dv in range(n_gpus):
                name = os.path.join(layer.__save_dir__, 'Batch{}_Device{}.pt'.format(b, dv))
                feature_map_norm.append(torch.load(name, map_location='cpu')['middle'].detach())
        layer.__feature_map_norm__ = torch.cat(feature_map_norm, dim=0).mean(dim=0)

        if len(save_dir) > 0:
            # s, _ = layer.__feature_map_norm__.sort()
            # s = s.numpy()
            s = (layer.__feature_map_norm__ / layer.__feature_map_norm__.max()).numpy()
            if not os.path.exists(os.path.join(os.path.dirname(save_dir), 'channel_norm_dist')):
                os.makedirs(os.path.join(os.path.dirname(save_dir), 'channel_norm_dist'))
            x = list(range(1, s.shape[0] + 1))
            plt.plot(x, s, 'b')
            plt.grid()
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Feature map norm distribuation in Layer {}'.format(l))
            plt.savefig(os.path.join(os.path.dirname(save_dir), 'channel_norm_dist/norm_l{}.png'.format(l)))
            plt.close()

def feat_norm_dist(save_dir, s, n, l):

    if not os.path.exists(os.path.join(os.path.dirname(save_dir), 'channel_norm_dist')):
        os.makedirs(os.path.join(os.path.dirname(save_dir), 'channel_norm_dist'))
    x = list(range(1, s.shape[0] + 1))
    plt.plot(x, s, 'b')
    plt.plot(n, s[n-1], 'ro')
    plt.grid()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Singular value distribuation in Layer {}'.format(l))
    plt.savefig(os.path.join(os.path.dirname(save_dir), 'channel_norm_dist/norm_l{}.png'.format(l)))
    plt.close()

    s_percent = s.cumsum() / s.sum()
    plt.plot(x, 100 * s_percent, 'b')
    plt.plot(n, 100 * s_percent[n-1], 'ro')
    plt.grid()
    plt.xlabel('Index')
    plt.ylabel('Percentage %')
    plt.title('Cumulative energy of singular value in Layer {}'.format(l))
    plt.savefig(os.path.join(os.path.dirname(save_dir), 'channel_norm_dist/percent_l{}.png'.format(l)))
    plt.close()

