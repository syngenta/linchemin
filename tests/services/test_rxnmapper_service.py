from linchemin.services.rxnmapper import service


def test_endpoint_metadata():
    rxnmapper_service = service.RxnMapperService(base_url='http://127.0.0.1:8002/')
    print("\n metadata")
    endpoint = rxnmapper_service.endpoint_map.get('metadata')
    inp_example = endpoint.input_example
    out_example = endpoint.output_example
    out_request = endpoint.submit(request_input=inp_example)
    print('input', inp_example)
    print('expected', out_example)
    print('actual', out_request)
    assert inp_example is None
    assert out_example.keys() == out_request.keys()


def test_endpoint_run_batch():
    rxnmapper_service = service.RxnMapperService(base_url='http://127.0.0.1:8002/')
    print("\n run_batch")
    endpoint = rxnmapper_service.endpoint_map.get('run_batch')
    inp_example = endpoint.input_example
    out_example = endpoint.output_example
    out_request = endpoint.submit(request_input=inp_example)
    print('input', inp_example)
    print('expected', out_example)
    print('actual', out_request)
    assert out_example.get('output') == out_request.get('output')
