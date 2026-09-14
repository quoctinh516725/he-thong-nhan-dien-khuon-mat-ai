import hashlib
def convert_uuid_to_int(uuid):
    return int.from_bytes(
        hashlib.sha256(str(uuid).encode()).digest()[:8],
        'big',
        signed=True
    )

