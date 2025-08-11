### Response Template ###
def en_response_scheme_template():
    return """
    Respond in JSON. Your response should be in the following format:
    {
      "label": "The label you assigned to the text segment.",
      "confidence": "A float value on a continuous scale from 0.0 to 1.0 that indicates how confident you are in your prediction. 0.0 means your are just guessing and 1.0 means you are absolut certain.",
      "explanation": "Explain why you assigned the label to the input."
    }
    """


def de_response_scheme_template():
    return """
    Antworte in JSON. Die Antwort soll folgendermaßen strukturiert sein:
    {
      "label": "Das Label, dass du einem Segment zuweist."
    }
    """
#"confidence": ""Ein Gleitkommawert auf einer kontinuierlichen Skala von 0.0 bis 1.0, der angibt, wie sicher Du dir bei deiner Vorhersage bist. 0.0 bedeutet, dass du nur rätst, und 1.0 bedeutet, dass Du dir absolut sicher bist.",
#"explanation": "Erläutere, warum du dem Segment dieses Label zugewiesen hast.

### System Prompt Template ###
def en_system_prompt_template(annotation_scheme_name, class_names):
    return f'You are a classifier that assigns labels to text segments.' \
           f'You label every segment with respect to {annotation_scheme_name}. ' \
           f'To assign the label chose one of the following categories:{", ".join(class_names)}.'


def de_system_prompt_template(annotation_scheme_name, class_names):
    return f'Du bist ein Klassifikator, der Textsegmenten Label zuweist.'\
           f'Du beschriftest jedes Label in Bezug auf {annotation_scheme_name}. ' \
           f'Um das Label zu vergeben, wählst du eine der folgenden Kategorien: {", ".join(class_names)}.'


### Description Template ###
def en_description_template(description):
    return f'Follow the following description of the labels:\n{description}'


def de_description_template(description):
    return f'Beachte die folgende Beschreibung der Label:\n{description}'


### Example template###
def en_example_template(examples):
    return f'Use the following examples as a guideline for your labeling process:\n {examples}'


def de_example_template(examples):
    return f'Verwende die folgenden Beispiele als Leitfaden:\n {examples}'


### Date and Message ###
def  en_message_template(sample, main_role):
    if main_role:
        return f'Label the following segment with respect to {main_role}:\n{sample}\nLabel:\n'
    else:
        return f'Label the following segment:\n{sample}\nLabel:\n'
def  de_message_template(sample, main_role):
    if main_role:
        return f'Labele das folgende Textsegment im Bezug auf {main_role}:\n{sample}\nLabel:\n'
    else:
        return f'Labele das folgende Textsegment:\n{sample}\nLabel:\n'


### Abstraction ###
def system_prompt_template(lang, annotation_scheme_name, class_names):
    if lang == 'en':
        return en_system_prompt_template(annotation_scheme_name, class_names)
    elif lang == 'de':
        return de_system_prompt_template(annotation_scheme_name, class_names)


def description_template(lang, description):
    if lang == 'en':
        return en_description_template(description)
    elif lang == 'de':
        return de_description_template(description)


def response_scheme_template(lang):
    if lang == 'en':
        return en_response_scheme_template()
    elif lang == 'de':
        return de_response_scheme_template()


def example_template(lang, description):
    if lang == 'en':
        return en_description_template(description)
    elif lang == 'de':
        return de_description_template(description)

def message_template(lang, sample, main_role=''):
    if lang == 'en':
        return en_message_template(sample, main_role)
    elif lang == 'de':
        return de_message_template(sample, main_role)
