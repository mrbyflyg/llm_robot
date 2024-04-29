import openai


def load_instruction_file(fpath):
    """
    Open the txt file for reading
    @param: fpath, the file path
    @return: the file contents
    """
    try:
        with open(fpath, "r") as file:
            # Read the contents of the file
            file_contents = file.read()
            # Process or print the file contents
            return file_contents

    except FileNotFoundError:
        print(f"File not found: {fpath}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def llm_call(messages, temperature=0.2, model="gpt-3.5-turbo"):
    """
    Call the OpenAI API to generate the response
    @param: messages, the conversation messages:
        {"role": "system", "content": motion_descriptor}
    @param: temperature, the temperature for the model
    @param: model, the model to use
    @return: the response from the model
    """
    client = openai.Client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response
