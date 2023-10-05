def to_cuda(data, device = 'cuda:0'):
    if type(data)==list:
        results = []
        for i, item in enumerate(data):
            if type(item).__name__ == "Tensor":
                results.append(item.to(device))
            elif type(item).__name__ == 'list':
                tensor_list = []
                for tensor in item:
                    tensor_list.append(tensor.to(device))
                results.append(tensor_list)
            else:
                raise NotImplementedError
        return results
    elif type(data)==dict:
        results={}
        for k,v in data.items():
            if type(v).__name__ == "Tensor":
                results[k]=v.to(device)
            elif type(v).__name__ == 'list':
                tensor_list = []
                for tensor in v:
                    tensor_list.append(tensor.to(device))
                results[k]=tensor_list
            else:
                raise NotImplementedError
        return results
    else:
        raise NotImplementedError