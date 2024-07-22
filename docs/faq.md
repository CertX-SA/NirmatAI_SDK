# Frequently Asked Questions (FAQ)

This document provides answers to some of the most frequently asked questions about the NirmatAI SDK.

## General Questions

### What is the NirmatAI SDK?

The NirmatAI SDK is a software development kit that provides tools and utilities to help you integrate with NirmatAI services.

### How do I install the SDK?

You can install the SDK using pip:


pip install NirmatAI


## Usage Questions

### How do I use the greet function?

The greet function generates a greeting message. Here's an example:


from NirmatAI import greet

message = greet("Alice")
print(message)  # Output: Hello, Alice!


### How do I use the add function?

The add function adds two integers. Here's an example:


from NirmatAI import add

sum_result = add(10, 20)
print(sum_result)  # Output: 30


## Troubleshooting

### I encountered an error while using the SDK. What should I do?

If you encounter any issues, please check the following:
- Ensure you have installed all required dependencies.
- Refer to the [user guide](user_guide.md) and [API reference](api_reference.md) for correct usage.
- Check the [examples](../examples/README.md) for practical usage scenarios.

If the problem persists, please open an issue on our [GitHub repository](https://github.com/CertX-SA/NirmatAI_SDK) with detailed information about the error.

### How can I contribute to the SDK?

We welcome contributions! Please refer to the CONTRIBUTING.md file in the root of the repository for guidelines on how to contribute.

For any additional questions, feel free to reach out to us through our [GitHub repository](https://github.com/CertX-SA/NirmatAI_SDK).