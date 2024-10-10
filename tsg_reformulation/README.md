## Converting TSGs into Structured Format
1. Reformulating Raw TSGs
    
    If your TSGs are not well-formulated, follow the step to convert raw TSGs into a structured JSON format.

    ```bash
    python tsg_reformulation.py
    ```
    This script will convert raw TSGs into a structured format, categorizing them into 'Terminology', 'Background', 'FAQ', 'STEPs', and 'Appendix'.

2. Extracting Nodes from Reformulated TSGs

    If you already have well-formulated TSGs, you can directly extract nodes and organize them into JSON format.
    ```bash
    python tsg_element.py
    ```

    This script will extract nodes from reformulated TSGs. Each node will include fields such as 'type', 'title', 'intent', 'action', 'output', and 'default_parameters'.

    Note: If you manually reformulate the TSGs, you directly use it to extract the element.

----

Below is an examplar structure to show the demand of clarity, organization and thoroughness. 

```markdown
# Background
## {question}
{background}

# Terminology
## {terminlogy}
{terminlogy explanation}

# FAQ
## {question}
{answer}

# Appendix
## {appendix title}
{appendix context}

# How to {investigate *** incident}? (STEPs)

## 1. {step1 title}
### Intent
intent
### Action
action (including necessary code)
### Linker
linker (must be in "if-then" format and "then" will directly triggers the intent of next step)

## 2. {step2 title}
### Intent
intent
### Action
action
### Linker
linker 
```
---
