### TOD specific LLM judge dimensions ###
mwoz_autotod_prompt_conv_consistency = """
Evaluate the **conversation consistency** of the following task-oriented dialogue chatbot response on a 5-point scale from "Very Bad" to "Very Good". 
The prompt will include the **dialogue history**, **current user query**, **database results**, **chatbot response**. 

## Conversation Consistency Definition
Conversation consistency refers to the degree to which the chatbot's response aligns with the context of the conversation, including:
- **Relevance**: The response directly relates to the dialogue history and the current user query.
- **Topic Consistency**: The response remains on-topic with the dialogue history and the user query.
- **Coherence**: The response logically continues the progression of the dialogue.

## Scoring Guide:
- **Very Good (5)**: Response is completely consistent with the previous conversation context, with no inconsistencies or errors.
- **Good (4)**: Response is mostly consistent with the context. Only minor improvements are needed.
- **Fair (3)**: Response is somewhat consistent but contains noticeable inconsistencies or lacks depth in addressing the context.
- **Bad (2)**: Response shows limited consistency with the conversation context and requires significant improvement.
- **Very Bad (1)**: Response is incoherent or completely inconsistent with the conversation context.

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Evaluate the conversation consistency of the following dialogue response
### Dialogue History
{dialogue_history}
### User Query
{user_query}
### Chatbot Response
{agent_response}
"""

mwoz_autotod_prompt_backend_consistency = """
Evaluate the **backend knowledge consistency** of the following task-oriented dialogue chatbot response on a 5-point scale from "Very Bad" to "Very Good". 
The prompt will include the **dialogue history**, **current user query**, **database results**, **chatbot response**. 

## Backend Knowledge Consistency Definition
Backend knowledge consistency refers to how well the chatbot's response aligns with the information provided in the policy or database results, considering:
- **Accuracy**: The response accurately reflects the information in the database results.
- **Topic Consistency**: The response stays on-topic with the database results and the dialogue context.
- **Coherence**: The response logically incorporates and progresses based on the database results.

## Scoring Guide:
- **Very Good (5)**: Response is completely consistent with the database results, with no inconsistencies or errors.
- **Good (4)**: Response is mostly consistent with the database results. Only minor improvements are needed.
- **Fair (3)**: Response is sufficiently consistent with the database results but contains noticeable inconsistencies or lacks depth in addressing the results.
- **Bad (2)**: Response shows limited consistency with the database results and requires significant improvement.
- **Very Bad (1)**: Response is incoherent or completely inconsistent with the database results.

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Evaluate the backend knowledge consistency of the following dialogue response
### Dialogue History
{dialogue_history}
### User Query
{user_query}
### Database Results
{db_result}
### Chatbot Response
{agent_response}
"""

# Policy Completeness split by domain
mwoz_autotod_prompt_restaurant_policy_compliance = """
Evaluate the **policy compliance** of the following task-oriented dialogue chatbot response on a 5-point scale from "Very Bad" to "Very Good". 
The prompt will include the **policy protocol**, **dialogue history**, **current user query**, **database results**, **chatbot response**. 

## Policy Compliance Definition
Policy compliance refers to how well the chatbot adheres to the expected policy protocol, specifically:
- **Number of Suggestions**: Providing suggestions only when the database results are few enough to do so.
- **Information Gathering**: Requesting required, relevant information (slots) from the user before offering suggestions or booking services.
- **Appropriate Timing**: Avoiding premature actions, such as making a booking or suggesting a service too early in the conversation, before the necessary information is gathered from the user.
- **Alignment with Policy**: Avoiding actions that do not align with the suggested flow of interaction in the policy, when available.

## Domain Possible Slots (May not be exhaustive)
The current predicted domain for this turn in the conversation is "Restaurant". You should check if domain-relevant slots have been filled in the current conversation. The possible slots for all domains are shown below (these may not be totally exhaustive):

### Restaurant
- area: the area where the restaurant is located
- bookday: the day for the reservation at the restaurant
- bookpeople: the number of people included in the restaurant reservation
- booktime: the time for the reservation at the restaurant
- food: the type of cuisine the restaurant serves
- name: the name of the restaurant
- pricerange: the price range of the restaurant

## Scoring Guide:
- **5 (Very Good)**: Response fully follows policy protocol with no errors or omissions.
- **4 (Good)**: Response mostly follows policy protocol, with only minor room for improvement.
- **3 (Fair)**: Response sufficiently follows policy protocol but has clear areas where it could improve in completeness or timing.
- **2 (Bad)**: Response does not adequately follow policy protocol, though there may be partial adherence.
- **1 (Very Bad)**: Response fails to follow policy protocol and is incomplete or incoherent.

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Policy Protocol
{policy}

## Evaluate the policy completeness of the following dialogue response
### Dialogue History
{dialogue_history}
### User Query
{user_query}
### Database Results
{db_result}
### Chatbot Response
{agent_response}
"""

mwoz_autotod_prompt_hotel_policy_compliance = """
Evaluate the **policy compliance** of the following task-oriented dialogue chatbot response on a 5-point scale from "Very Bad" to "Very Good". 
The prompt will include the **policy protocol**, **dialogue history**, **current user query**, **database results**, **chatbot response**. 

## Policy Compliance Definition
Policy compliance refers to how well the chatbot adheres to the expected policy protocol, specifically:
- **Number of Suggestions**: Providing suggestions only when the database results are few enough to do so.
- **Information Gathering**: Requesting required, relevant information (slots) from the user before offering suggestions or booking services.
- **Appropriate Timing**: Avoiding premature actions, such as making a booking or suggesting a service too early in the conversation, before the necessary information is gathered from the user.
- **Alignment with Policy**: Avoiding actions that do not align with the suggested flow of interaction in the policy, when available.

## Domain Possible Slots (May not be exhaustive)
The current predicted domain for this turn in the conversation is "Hotel". You should check if domain-relevant slots have been filled in the current conversation. The possible slots for all domains are shown below (these may not be totally exhaustive):

### Hotel
- area: the area where the hotel is located
- bookday: the day of the booking for the hotel
- bookpeople: the number of people included in the hotel booking
- bookstay: the duration of the stay at the hotel (e.g., number of nights)
- internet: whether the hotel offers internet or Wi-Fi (true/false)
- name: the name of the hotel
- parking: whether the hotel provides parking facilities (true/false)
- pricerange: the price range of the hotel (e.g., cheap, moderate, expensive)
- stars: the star rating of the hotel
- type: the type of the hotel (e.g., hotel, bed and breakfast, guest house)

## Scoring Guide:
- **5 (Very Good)**: Response fully follows policy protocol with no errors or omissions.
- **4 (Good)**: Response mostly follows policy protocol, with only minor room for improvement.
- **3 (Fair)**: Response sufficiently follows policy protocol but has clear areas where it could improve in completeness or timing.
- **2 (Bad)**: Response does not adequately follow policy protocol, though there may be partial adherence.
- **1 (Very Bad)**: Response fails to follow policy protocol and is incomplete or incoherent.

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Policy Protocol
{policy}

## Evaluate the policy completeness of the following dialogue response
### Dialogue History
{dialogue_history}
### User Query
{user_query}
### Database Results
{db_result}
### Chatbot Response
{agent_response}
"""

mwoz_autotod_prompt_attraction_policy_compliance = """
Evaluate the **policy compliance** of the following task-oriented dialogue chatbot response on a 5-point scale from "Very Bad" to "Very Good". 
The prompt will include the **policy protocol**, **dialogue history**, **current user query**, **database results**, **chatbot response**. 

## Policy Compliance Definition
Policy compliance refers to how well the chatbot adheres to the expected policy protocol, specifically:
- **Number of Suggestions**: Providing suggestions only when the database results are few enough to do so.
- **Information Gathering**: Requesting required, relevant information (slots) from the user before offering suggestions or booking services.
- **Appropriate Timing**: Avoiding premature actions, such as making a booking or suggesting a service too early in the conversation, before the necessary information is gathered from the user.
- **Alignment with Policy**: Avoiding actions that do not align with the suggested flow of interaction in the policy, when available.

## Domain Possible Slots (May not be exhaustive)
The current predicted domain for this turn in the conversation is "Attraction". You should check if domain-relevant slots have been filled in the current conversation. The possible slots for all domains are shown below (these may not be totally exhaustive):

### Attraction
- area: the area where the attraction is located
- type: the type of attraction (e.g., museum, gallery, theatre, concert, stadium)
- name: the name of the specific attraction being searched for

## Scoring Guide:
- **5 (Very Good)**: Response fully follows policy protocol with no errors or omissions.
- **4 (Good)**: Response mostly follows policy protocol, with only minor room for improvement.
- **3 (Fair)**: Response sufficiently follows policy protocol but has clear areas where it could improve in completeness or timing.
- **2 (Bad)**: Response does not adequately follow policy protocol, though there may be partial adherence.
- **1 (Very Bad)**: Response fails to follow policy protocol and is incomplete or incoherent.

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Policy Protocol
{policy}

## Evaluate the policy completeness of the following dialogue response
### Dialogue History
{dialogue_history}
### User Query
{user_query}
### Database Results
{db_result}
### Chatbot Response
{agent_response}
"""

mwoz_autotod_prompt_train_policy_compliance = """
Evaluate the **policy compliance** of the following task-oriented dialogue chatbot response on a 5-point scale from "Very Bad" to "Very Good". 
The prompt will include the **policy protocol**, **dialogue history**, **current user query**, **database results**, **chatbot response**. 

## Policy Compliance Definition
Policy compliance refers to how well the chatbot adheres to the expected policy protocol, specifically:
- **Number of Suggestions**: Providing suggestions only when the database results are few enough to do so.
- **Information Gathering**: Requesting required, relevant information (slots) from the user before offering suggestions or booking services.
- **Appropriate Timing**: Avoiding premature actions, such as making a booking or suggesting a service too early in the conversation, before the necessary information is gathered from the user.
- **Alignment with Policy**: Avoiding actions that do not align with the suggested flow of interaction in the policy, when available.

## Domain Possible Slots (May not be exhaustive)
The current predicted domain for this turn in the conversation is "Train". You should check if domain-relevant slots have been filled in the current conversation. The possible slots for all domains are shown below (these may not be totally exhaustive):

### Train
- arriveby: the time by which the train should arrive at the destination
- bookpeople: the number of people traveling on the train
- day: the day of the train journey (e.g., Monday, Tuesday, etc.)
- departure: the station from which the train departs
- destination: the station where the train journey ends
- leaveat: the time when the train should depart from the departure station

## Scoring Guide:
- **5 (Very Good)**: Response fully follows policy protocol with no errors or omissions.
- **4 (Good)**: Response mostly follows policy protocol, with only minor room for improvement.
- **3 (Fair)**: Response sufficiently follows policy protocol but has clear areas where it could improve in completeness or timing.
- **2 (Bad)**: Response does not adequately follow policy protocol, though there may be partial adherence.
- **1 (Very Bad)**: Response fails to follow policy protocol and is incomplete or incoherent.

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Policy Protocol
{policy}

## Evaluate the policy completeness of the following dialogue response
### Dialogue History
{dialogue_history}
### User Query
{user_query}
### Database Results
{db_result}
### Chatbot Response
{agent_response}
"""

mwoz_autotod_prompt_taxi_policy_compliance = """
Evaluate the **policy compliance** of the following task-oriented dialogue chatbot response on a 5-point scale from "Very Bad" to "Very Good". 
The prompt will include the **policy protocol**, **dialogue history**, **current user query**, **database results**, **chatbot response**. 

## Policy Compliance Definition
Policy compliance refers to how well the chatbot adheres to the expected policy protocol, specifically:
- **Number of Suggestions**: Providing suggestions only when the database results are few enough to do so.
- **Information Gathering**: Requesting required, relevant information (slots) from the user before offering suggestions or booking services.
- **Appropriate Timing**: Avoiding premature actions, such as making a booking or suggesting a service too early in the conversation, before the necessary information is gathered from the user.
- **Alignment with Policy**: Avoiding actions that do not align with the suggested flow of interaction in the policy, when available.

## Domain Possible Slots (May not be exhaustive)
The current predicted domain for this turn in the conversation is "Taxi". You should check if domain-relevant slots have been filled in the current conversation. The possible slots for all domains are shown below (these may not be totally exhaustive):

### Taxi
- arriveby: the time by which the taxi should arrive at the destination
- departure: the location where the taxi ride begins
- destination: the location where the taxi ride ends
- leaveat: the time when the taxi should pick up passengers

## Scoring Guide:
- **5 (Very Good)**: Response fully follows policy protocol with no errors or omissions.
- **4 (Good)**: Response mostly follows policy protocol, with only minor room for improvement.
- **3 (Fair)**: Response sufficiently follows policy protocol but has clear areas where it could improve in completeness or timing.
- **2 (Bad)**: Response does not adequately follow policy protocol, though there may be partial adherence.
- **1 (Very Bad)**: Response fails to follow policy protocol and is incomplete or incoherent.

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Policy Protocol
{policy}

## Evaluate the policy completeness of the following dialogue response
### Dialogue History
{dialogue_history}
### User Query
{user_query}
### Database Results
{db_result}
### Chatbot Response
{agent_response}
"""

prompt_mwoz_autotod_unknown_policy_compliance = """
Evaluate the **policy compliance** of the following task-oriented dialogue chatbot response on a 5-point scale from "Very Bad" to "Very Good". 
The prompt will include the **policy protocol**, **dialogue history**, **current user query**, **database results**, **chatbot response**. 

## Policy Compliance Definition
Policy compliance refers to how well the chatbot adheres to the expected policy protocol, specifically:
- **Number of Suggestions**: Providing suggestions only when the database results are few enough to do so.
- **Information Gathering**: Requesting required, relevant information (slots) from the user before offering suggestions or booking services.
- **Appropriate Timing**: Avoiding premature actions, such as making a booking or suggesting a service too early in the conversation, before the necessary information is gathered from the user.
- **Alignment with Policy**: Avoiding actions that do not align with the suggested flow of interaction in the policy, when available.

## Scoring Guide:
- **5 (Very Good)**: Response fully follows policy protocol with no errors or omissions.
- **4 (Good)**: Response mostly follows policy protocol, with only minor room for improvement.
- **3 (Fair)**: Response sufficiently follows policy protocol but has clear areas where it could improve in completeness or timing.
- **2 (Bad)**: Response does not adequately follow policy protocol, though there may be partial adherence.
- **1 (Very Bad)**: Response fails to follow policy protocol and is incomplete or incoherent.

## Policy Protocol
{policy}

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Evaluate the policy completeness of the following dialogue response
### Dialogue History
{dialogue_history}
### User Query
{user_query}
### Database Results
{db_result}
### Chatbot Response
{agent_response}
"""

## DIAL LEVEL PROMPTS ##
autotod_dial_level_prompt_conv_consistency = """
Evaluate the **overall conversation consistency** of the following task-oriented dialogue chatbot conversation on a 5-point scale from "Very Bad" to "Very Good". 
The conversation will include **user queries**, **database results**, **chatbot responses**. 

## Conversation Consistency Definition
Conversation consistency refers to the degree to which the chatbot's responses aligns with the context of the conversation, including:
- **Relevance**: The response directly relates to the dialogue history and the current user query.
- **Topic Consistency**: The response remains on-topic with the dialogue history and the user query.
- **Coherence**: The response logically continues the progression of the dialogue.

## Scoring Guide:
- **Very Good (5)**: Responses are completely consistent with the previous conversation context, with no inconsistencies or errors.
- **Good (4)**: Responses are mostly consistent with the context. Only minor improvements are needed.
- **Fair (3)**: Responses are somewhat consistent but contains noticeable inconsistencies or lacks depth in addressing the context.
- **Bad (2)**: Responses show limited consistency with the conversation context and requires significant improvement.
- **Very Bad (1)**: Responses are incoherent or completely inconsistent with the conversation context.

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Evaluate the conversation consistency of the following dialogue response
### Full Dialogue
{dialogue_history}
"""

autotod_dial_level_prompt_backend_consistency = """
Evaluate the **overall backend knowledge consistency** of the following task-oriented dialogue chatbot conversation on a 5-point scale from "Very Bad" to "Very Good". 
The prompt will include the **user queries**, **database results**, **chatbot responses**. 

## Backend Knowledge Consistency Definition
Backend knowledge consistency refers to how well the chatbot's responses aligns with the information provided in the policy or database results, considering:
- **Accuracy**: The response accurately reflects the information in the database results.
- **Topic Consistency**: The response stays on-topic with the database results and the dialogue context.
- **Coherence**: The response logically incorporates and progresses based on the database results.

## Scoring Guide:
- **Very Good (5)**: Response is completely consistent with the database results, with no inconsistencies or errors.
- **Good (4)**: Response is mostly consistent with the database results. Only minor improvements are needed.
- **Fair (3)**: Response is sufficiently consistent with the database results but contains noticeable inconsistencies or lacks depth in addressing the results.
- **Bad (2)**: Response shows limited consistency with the database results and requires significant improvement.
- **Very Bad (1)**: Response is incoherent or completely inconsistent with the database results.

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Evaluate the backend knowledge consistency of the following dialogue response
### Full Dialogue
{dialogue_history}
"""

# Policy Completeness split by domain
autotod_dial_level_prompt_policy_compliance = """
Evaluate the **overall policy compliance** of the following task-oriented dialogue chatbot conversation on a 5-point scale from "Very Bad" to "Very Good". 
The prompt will include the **policy protocol**, **user queries**, **database results**, **chatbot responses**. 

## Policy Compliance Definition
Policy compliance refers to how well the chatbot adheres to the expected policy protocol, specifically:
- **Number of Suggestions**: Providing suggestions only when the database results are few enough to do so.
- **Information Gathering**: Requesting required, relevant information (slots) from the user before offering suggestions or booking services.
- **Appropriate Timing**: Avoiding premature actions, such as making a booking or suggesting a service too early in the conversation, before the necessary information is gathered from the user.
- **Alignment with Policy**: Avoiding actions that do not align with the suggested flow of interaction in the policy, when available.

## Scoring Guide:
- **5 (Very Good)**: Response fully follows policy protocol with no errors or omissions.
- **4 (Good)**: Response mostly follows policy protocol, with only minor room for improvement.
- **3 (Fair)**: Response sufficiently follows policy protocol but has clear areas where it could improve in completeness or timing.
- **2 (Bad)**: Response does not adequately follow policy protocol, though there may be partial adherence.
- **1 (Very Bad)**: Response fails to follow policy protocol and is incomplete or incoherent.

## Policy Protocol
{policy}

## Always include the score first, then a rationale on a new line. The rationale should be at most 2 sentences. Follow the template below:
Score: [YOUR SCORE NUMBER HERE]
Justification: [YOUR RATIONALE HERE]

## Evaluate the policy completeness of the following dialogue response
### Full Dialogue
{dialogue_history}
"""