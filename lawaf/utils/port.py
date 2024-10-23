import socket

def get_unused_port(default=8000):
    """
    Get an unused port number
    """
    if default is not None and not is_port_in_use(default) :
        return default
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def is_port_in_use(port):
    """
    Check if a port is in use
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def test_is_port_in_use():
    """
    Test the is_port_in_use function to ensure it correctly identifies if a port is in use.
    """
    assert is_port_in_use(22), "Port 22 should be in use"
    assert not is_port_in_use(12345), "Port 12345 should not be in use"

def test_get_unused_port():
    """
    Test the get_unused_port function to ensure it returns a valid, unused port number.
    """
    port = get_unused_port()
    assert isinstance(port, int), "Port should be an integer"
    assert 1024 <= port <= 65535, "Port should be in the valid range for non-privileged ports"
    assert not is_port_in_use(port), "Port should not be in use"

if __name__ == "__main__":
    test_is_port_in_use()
    test_get_unused_port()
    print("port.py: All tests passed")
