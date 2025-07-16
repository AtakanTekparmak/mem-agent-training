def serialize_tilde(strings: list[str]) -> str:
    """
    Serialize a list of strings into a single string using '~@~' as delimiter.
    
    Args:
        strings: List of strings to serialize
        
    Returns:
        A single string with elements joined by '~@~'
    """
    return "~@~".join(strings)


def deserialize_tilde(data: str) -> list[str]:
    """
    Deserialize a string back into a list of strings using '~@~' as delimiter.
    
    Args:
        data: String containing elements separated by '~@~'
        
    Returns:
        List of strings split by the delimiter
    """
    if not data:
        return []
    return data.split("~@~")
