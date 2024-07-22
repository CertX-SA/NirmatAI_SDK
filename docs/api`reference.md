# API Reference

This document provides a detailed reference for the NirmatAI SDK.

## Modules

### NirmatAI.core

#### greet(name: str) -> str

Generate a greeting message.

- Parameters: name (str) – The name of the person to greet.
- Returns: str – Greeting message.

Example:


from NirmatAI.core import greet

message = greet("Alice")
print(message)  # Output: Hello, Alice!


### NirmatAI.utils

#### add(a: int, b: int) -> int

Add two integers.

- Parameters:
  - a (int) – First integer.
  - b (int) – Second integer.
- Returns: int – The sum of the two integers.

Example:


from NirmatAI.utils import add

sum_result = add(10, 20)
print(sum_result)  # Output: 30


For additional details and usage examples, please refer to the [user guide](user_guide.md) and the [examples](../examples) directory.
