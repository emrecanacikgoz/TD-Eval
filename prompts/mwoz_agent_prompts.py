## Part 1: Domain Classification ##
mwz_domain_prompt = """
Determine which domain is considered in the following dialogue situation.
Choose one domain from this list:
 - restaurant
 - hotel
 - attraction
 - taxi
 - train
Answer with only one word, the selected domain from the list.
You have to always select the closest possible domain.
Consider the last domain mentioned, so focus mainly on the last utterance.
-------------------
Example 1:
history:
---
Customer: I need a cheap place to eat
Assistant: We have several not expensive places available. What food are you interested in?
---
Customer: Chinese food.
---
Domain: restaurant
------
Example 2:
history:
---
Customer: I also need a hotel in the north.
Assistant: Ok, can I offer you the Molly's place?
---
Customer: What is the address?
---
Domain: hotel
-------------------
Now complete the following example:
history: 
---
{history}
---
Customer: {utterance}
---
Domain:"""

## Part 2: State Tracking ##
# TODO: follow state tracking prompt in old ConvAI repo
mwz_restaurant_state_prompt = """
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair 'entity':'value' separated by colon and no spaces in between.
Separate 'entity':'value' pairs by hyphens. The entities and values should be closed by single quotes.
Values that should be captured are:
 - pricerange: price range of the restaurant (cheap/moderate/expensive)
 - area: area where the restaurant is located (north/east/west/south/centre)
 - bookday: day of week reservation booked (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - bookpeople: number of people booked (1/2/3...)
 - food: type of food served by restaurant
 - name: the name of restaurant
 - booktime: time of booked reservation by hour and minute (i.e. 15:30)
Do not capture any other values! If not specified, leave the value empty.
---
history:
{history}
---
Customer: {utterance}
---
state: """

mwz_hotel_state_prompt = """
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair 'entity':'value' separated by colon and no spaces in between.
Separate 'entity':'value' pairs by hyphens. The entities and values should be closed by single quotes.
Values that should be captured are:
 - pricerange: the price range of the hotel (cheap/expensive)
 - parking: if the hotel has parking (yes/no)
 - internet: if the hotel has internet (yes/no)
 - stars: the number of stars the hotel has (1/2/3/4/5)
 - area: the area where the hotel is located (north/east/west/south/centre)
 - type: the type of the hotel (hotel/bed and breakfast/guest house)
 - bookpeople: number of people booked (1/2/3...)
 - bookday: day of week reservation booked (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - bookstay: how many nights booked (1/2/3...)
 - name: the name of hotel
Do not capture any other values! If not specified, leave the value empty.
---
history:
{history}
---
Customer: {utterance}
---
state: """

mwz_attraction_state_prompt = """
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair 'entity':'value' separated by colon and no spaces in between.
Separate 'entity':'value' pairs by hyphens. The entities and values should be closed by single quotes.
Values that should be captured are:
 - area: that specifies the area where the attraction is located (north/east/west/south/centre)
 - type: that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - name: the name of the attraction
Do not capture any other values! If not specified, leave the value empty.
---
history:
{history}
---
Customer: {utterance}
---
state: """

mwz_train_state_prompt = """
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair 'entity':'value' separated by colon and no spaces in between.
Separate 'entity':'value' pairs by hyphens. The entities and values should be closed by single quotes.
Values that should be captured are:
 - departure: the departure station of train
 - destination: the destination station of train
 - day: what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - bookpeople: number of people booked for train (1/2/3...)
 - arriveBy: what time the train should arrive
 - leaveAt: what time the train should leave
Do not capture any other values! If not specified, leave the value empty.
---
history:
{history}
---
Customer: {utterance}
---
state: """

mwz_taxi_state_prompt = """
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair 'entity':'value' separated by colon and no spaces in between.
Separate 'entity':'value' pairs by hyphens. The entities and values should be closed by single quotes.
Values that should be captured are:
 - destination: taxi destination station
 - departure: taxi departure station
 - arriveBy: time the taxi should arrive
 - leaveAt: what time the taxi should leave
 Do not capture any other values! If not specified, leave the value empty.
---
history:
{history}
---
Customer: {utterance}
---
state: """

MWZ_DOMAIN_STATE_PROMPTS = {
  "restaurant": mwz_restaurant_state_prompt,
  "hotel": mwz_hotel_state_prompt,
  "attraction": mwz_attraction_state_prompt,
  "train": mwz_train_state_prompt,
  "taxi": mwz_taxi_state_prompt
}

## Part 3: Responses ##
mwz_restaurant_response_prompt = """
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or price.
There is also a number of restaurants in the database currently corresponding to the user's request.
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
The values that can be captured are:
 - pricerange: price range of the restaurant (cheap/moderate/expensive)
 - area: area where the restaurant is located (north/east/west/south/centre)
 - bookday: day of week reservation booked (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - bookpeople: number of people booked (1/2/3...)
 - food: type of food served by restaurant
 - name: the name of restaurant
 - booktime: time of booked reservation by hour and minute (i.e. 15:30)
 - address: address location of restaurant
 - phone: phone number of restaurant
 - postcode: postcode of restaurant
 - ref: reference number for reservation
If you find a restaurant, provide [restaurant_name], [restaurant_address], [restaurant_phone] or [restaurant_postcode] if asked.
If booking is successful,  provide an 8 digit alphanumeric reference code in the answer.
Write your response on only one line, no newlines or markdown formatting.
---
history:
{history}
---
Customer: {utterance}
---
state: {state}
---
database: {database}
---
response:"""

mwz_hotel_response_prompt = """
Definition: You are an assistant that helps people to book a hotel.
The customer can ask for a hotel by name, area, parking, internet availability, or price.
There is also a number of hotel in the database currently corresponding to the user's request.
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
The values that can be captured are:
 - pricerange: the price range of the hotel (cheap/expensive)
 - parking: if the hotel has parking (yes/no)
 - internet: if the hotel has internet (yes/no)
 - stars: the number of stars the hotel has (1/2/3/4/5)
 - area: the area where the hotel is located (north/east/west/south/centre)
 - type: the type of the hotel (hotel/bed and breakfast/guest house)
 - bookpeople: number of people booked (1/2/3...)
 - bookday: day of week reservation booked (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - bookstay: how many nights booked (1/2/3...)
 - name: the name of hotel
 - address: address location of hotel
 - phone: phone number of hotel
 - postcode: postcode of hotel
 - ref: reference number for reservation
If booking is successful,  provide an 8 digit alphanumeric reference code in the answer.
Write your response on only one line, no newlines or markdown formatting.
---
history:
{history}
---
Customer: {utterance}
---
state: {state}
---
database: {database}
---
response:"""

mwz_attraction_response_prompt = """
Definition: You are an assistant that helps people to find an attraction.
The customer can ask for an attraction by name, area, or type.
There is also a number of restaurants provided in the database.
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
The values that can be captured are:
 - area: that specifies the area where the attraction is located (north/east/west/south/centre)
 - type: that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - name: the name of the attraction
 - address: address location of attraction
 - entrancefee: entrance fee for attraction
 - openhours: opening hours of attraction
 - phone: phone number of attraction
 - postcode: postcode of attraction
If you find an attraction, provide [attraction_name], [attraction_address], [attraction_phone] or [attraction_postcode] if asked.
Write your response on only one line, no newlines or markdown formatting.
---
history:
{history}
---
Customer: {utterance}
---
state: {state}
---
database: {database}
---
response:"""

mwz_train_response_prompt = """
Definition: You are an assistant that helps people to find a train connection.
The customer needs to specify the departure and destination station, and the time of departure or arrival.
There is also a number of trains in the database currently corresponding to the user's request.
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
The values that can be captured are:
 - departure: the departure station of train
 - destination: the destination station of train
 - day: what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - bookpeople: number of people booked for train (1/2/3...)
 - arriveBy: what time the train should arrive
 - leaveAt: what time the train should leave
 - trainId: id of the train
 - ref: reference number for train ticket
 - price: price of train ticket
 - duration: how long train ride is
If you find a train, provide [arriveby], [leaveat] or [departure] if asked.
If booking is successful, provide an 8 digit alphanumeric reference code in the answer.
Write your response on only one line, no newlines or markdown formatting.
---
history:
{history}
---
Customer: {utterance}
---
state: {state}
---
database: {database}
---
response:"""

mwz_taxi_response_prompt = """
Definition: You are an assistant that helps people to book a taxi.
If the database returns a number, then there are too many possible items. You can ask for more information. 
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
The values that can be captured are:
 - destination: taxi destination station
 - departure: taxi departure station
 - arriveBy: time the taxi should arrive
 - leaveAt: what time the taxi should leave
 - phone: phone number for taxi service
 - type: color and make/model of taxi
If you find a taxi, provide [arriveby], [leaveat] or [departure] if asked.
If booking is successful, provide an 8 digit alphanumeric reference code in the answer.
Write your response on only one line, no newlines or markdown formatting.
---
history:
{history}
---
Customer: {utterance}
---
state: {state}
---
database: {database}
---
response:"""

MWZ_DOMAIN_RESPONSE_PROMPTS = {
  "restaurant": mwz_restaurant_response_prompt,
  "hotel": mwz_hotel_response_prompt,
  "attraction": mwz_attraction_response_prompt,
  "train": mwz_train_response_prompt,
  "taxi": mwz_taxi_response_prompt
}

mwz_restaurant_delex_prompt = """
You are an assistant that delexicalizes concrete entities into slot values. You will receive a response from a chatbot that helps people book a restaurant. Please replace relevant entities with the entity type in brackets. 
The entity types that can be delex-ed are:
 - pricerange: price range of the restaurant (cheap/moderate/expensive)
 - area: area where the restaurant is located (north/east/west/south/centre)
 - bookday: day of week reservation booked (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - bookpeople: number of people booked (1/2/3...)
 - food: type of food served by restaurant
 - name: the name of restaurant
 - booktime: time of booked reservation by hour and minute (i.e. 15:30)
 - address: address location of restaurant
 - phone: phone number of restaurant
 - postcode: postcode of restaurant
 - ref: reference code for reservation
---
response: {response}
---
delex: 
"""

mwz_hotel_delex_prompt = """
You are an assistant that delexicalizes concrete entities into slot values. You will receive a response from a chatbot that helps people book a hotel. Please replace relevant entities with the entity type in brackets. 
The entity types that can be delex-ed are:
 - pricerange: the price range of the hotel (cheap/expensive)
 - parking: if the hotel has parking (yes/no)
 - internet: if the hotel has internet (yes/no)
 - stars: the number of stars the hotel has (1/2/3/4/5)
 - area: the area where the hotel is located (north/east/west/south/centre)
 - type: the type of the hotel (hotel/bed and breakfast/guest house)
 - bookpeople: number of people booked (1/2/3...)
 - bookday: day of week reservation booked (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - bookstay: how many nights booked (1/2/3...)
 - name: the name of hotel
 - address: address location of hotel
 - phone: phone number of hotel
 - postcode: postcode of hotel
 - ref: reference code for reservation
---
response: {response}
---
delex: 
"""

mwz_attraction_delex_prompt = """
You are an assistant that delexicalizes concrete entities into slot values. You will receive a response from a chatbot that helps people look for local attractions. Please replace relevant entities with the entity type in brackets. 
The entity types that can be delex-ed are:
 - area: that specifies the area where the attraction is located (north/east/west/south/centre)
 - type: that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - name: the name of the attraction
 - address: address location of attraction
 - entrancefee: entrance fee for attraction
 - openhours: opening hours of attraction
 - phone: phone number of attraction
 - postcode: postcode of attraction
---
response: {response}
---
delex: 
"""

mwz_train_delex_prompt = """
You are an assistant that delexicalizes concrete entities into slot values. You will receive a response from a chatbot that helps people book for train rides. Please replace relevant entities with the entity type in brackets. 
The entity types that can be delex-ed are:
 - departure: the departure station of train
 - destination: the destination station of train
 - day: day of the week the train is leaving (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - bookpeople: number of people booked for train (1/2/3...)
 - arriveBy: what time the train should arrive
 - leaveAt: what time the train should leave
 - trainId: id of the train
 - ref: reference code for train ticket
 - price: price of train ticket
 - duration: how long train ride is
---
response: {response}
---
delex: 
"""

mwz_taxi_delex_prompt = """
You are an assistant that delexicalizes concrete entities into slot values. You will receive a response from a chatbot that helps people book a taxi. Please replace relevant entities with the entity type in brackets. 
The entity types that can be delex-ed are:
 - destination: taxi destination station
 - departure: taxi departure station
 - arriveBy: time the taxi should arrive
 - leaveAt: what time the taxi should leave
 - phone: phone number for taxi service
 - type: color and make/model of taxi
---
response: {response}
---
delex: 
"""

MWZ_DOMAIN_DELEX_PROMPTS = {
  "restaurant": mwz_restaurant_delex_prompt,
  "hotel": mwz_hotel_delex_prompt,
  "attraction": mwz_attraction_delex_prompt,
  "train": mwz_train_delex_prompt,
  "taxi": mwz_taxi_delex_prompt
}