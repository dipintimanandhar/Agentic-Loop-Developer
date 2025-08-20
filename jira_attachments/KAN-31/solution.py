def solve_story(text: str) -> str:
    """
    This function corresponds to the user story KAN-31.
    It converts a given string to its uppercase equivalent.

    Args:
        text: The input string to be converted.

    Returns:
        The uppercase version of the input string.

    Examples:
        >>> solve_story('hello')
        'HELLO'
        >>> solve_story('world')
        'WORLD'
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")
        
    return text.upper()

# Example usage (not part of the required solution, but for demonstration)
if __name__ == '__main__':
    input1 = 'hello'
    output1 = solve_story(input1)
    print(f"Input: '{input1}', Output: '{output1}'")

    input2 = 'world'
    output2 = solve_story(input2)
    print(f"Input: '{input2}', Output: '{output2}'")