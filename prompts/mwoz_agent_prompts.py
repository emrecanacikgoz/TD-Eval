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
Example1:
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
mwz_restaurant_state_prompt = """
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - pricerange: that specifies the price range of the restaurant (cheap/moderate/expensive)
 - area: that specifies the area where the restaurant is located (north/east/west/south/centre)
 - food: that specifies the type of food the restaurant serves
 - name: that is the name of the restaurant
Do not capture any other values! If not specified, leave the value empty.
Please structure your output in json format. {{}} is a valid output.
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
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - area: that specifies the area where the hotel is located (north/east/west/south/centre)
 - internet: that specifies if the hotel has internet (yes/no)
 - parking: that specifies if the hotel has parking (yes/no)
 - stars: that specifies the number of stars the hotel has (1/2/3/4/5)
 - type: that specifies the type of the hotel (hotel/bed and breakfast/guest house)
 - pricerange: that specifies the price range of the hotel (cheap/expensive)
Do not capture any other values! If not specified, leave the value empty.
Please structure your output in json format. {{}} is a valid output.
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
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - type: that specifies the type of attraction (museum/gallery/theatre/concert/stadium)
 - area: that specifies the area where the attraction is located (north/east/west/south/centre)
 - name: the name of the specific attraction being searched for
Do not capture any other values! If not specified, leave the value empty.
Please structure your output in json format. {{}} is a valid output.
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
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - arriveBy: that specifies what time the train should arrive
 - leaveAt: that specifies what time the train should leave
 - day: that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
 - departure: that specifies the departure station
 - destination: that specifies the destination station
Do not capture any other values! If not specified, leave the value empty.
Please structure your output in json format. {{}} is a valid output.
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
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Values that should be captured are:
 - arriveBy: that specifies what time the train should arrive
 - leaveAt: that specifies what time the train should leave
 - departure: that specifies the departure station
 - destination: that specifies the destination station
 - day: that specifies what day the train should leave (monday/tuesday/wednesday/thursday/friday/saturday/sunday)
Do not capture any other values! If not specified, leave the value empty.
Please structure your output in json format. {{}} is a valid output.
---
history:
{history}
---
Customer: {utterance}
---
state: """

mwz_hospital_state_prompt = """
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Do not capture any other values! If not specified, leave the value empty.
Please structure your output in json format. {{}} is a valid output.
---
history:
{history}
---
Customer: {utterance}
---
state: """

mwz_bus_state_prompt = """
Capture entity values from last utterance of the converstation according to examples.
Focus only on the values mentioned in the last utterance.
Capture pair "entity:value" separated by colon and no spaces in between.
Separate entity:value pairs by hyphens.
Do not capture any other values! If not specified, leave the value empty.
Please structure your output in json format. {{}} is a valid output.
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
  "taxi": mwz_taxi_state_prompt,
  "hospital": mwz_hospital_state_prompt,
  "bus": mwz_bus_state_prompt
}

## Part 3: Responses ##
mwz_restaurant_response_prompt = """
Definition: You are an assistant that helps people to book a restaurant.
You can search for a restaurant by area, food, or price.
There is also a number of restaurants in the database currently corresponding to the user's request.
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
The values that can be captured are:
  - area: the area where the restaurant is located
  - bookday: the day for the reservation at the restaurant
  - bookpeople: the number of people included in the restaurant reservation
  - booktime: the time for the reservation at the restaurant
  - food: the type of cuisine the restaurant serves
  - name: the name of the restaurant
  - pricerange: the price range of the restaurant
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
output:"""

mwz_hotel_response_prompt = """
Definition: You are an assistant that helps people to book a hotel.
The customer can ask for a hotel by name, area, parking, internet availability, or price.
There is also a number of hotel in the database currently corresponding to the user's request.
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
The values that can be captured are:
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
  If you find a hotel, provide [hotel_name], [hotel_address], [hotel_phone] or [hotel_postcode] if asked.
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
output:"""

mwz_attraction_response_prompt = """
Definition: You are an assistant that helps people to find an attraction.
The customer can ask for an attraction by name, area, or type.
There is also a number of restaurants provided in the database.
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
The values that can be captured are:
  - area: the area where the attraction is located
  - type: the type of attraction (e.g., museum, gallery, theatre, concert, stadium)
  - name: the name of the specific attraction being searched for
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
output:"""

mwz_train_response_prompt = """
Definition: You are an assistant that helps people to find a train connection.
The customer needs to specify the departure and destination station, and the time of departure or arrival.
There is also a number of trains in the database currently corresponding to the user's request.
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
The values that can be captured are:
  - arriveby: the time by which the train should arrive at the destination
  - bookpeople: the number of people traveling on the train
  - day: the day of the train journey (e.g., Monday, Tuesday, etc.)
  - departure: the station from which the train departs
  - destination: the station where the train journey ends
  - leaveat: the time when the train should depart from the departure station
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
output:"""

mwz_taxi_response_prompt = """
Definition: You are an assistant that helps people to book a taxi.
If the database returns a number, then there are too many possible items. You can ask for more information. 
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
The values that can be captured are:
  - arriveby: the time by which the taxi should arrive at the destination
  - departure: the location where the taxi ride begins
  - destination: the location where the taxi ride ends
  - leaveat: the time when the taxi should pick up passengers
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
output:"""

mwz_hospital_response_prompt = """
Definition: You are an assistant that helps people to find a hospital.
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
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
output:response:"""
                                                
mwz_bus_response_prompt = """
Definition: You are an assistant that helps people to find a bus.
If the database results only return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results. 
If the database results also return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide one or some relevant entries to the user
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
output:response:"""

MWZ_DOMAIN_RESPONSE_PROMPTS = {
  "restaurant": mwz_restaurant_response_prompt,
  "hotel": mwz_hotel_response_prompt,
  "attraction": mwz_attraction_response_prompt,
  "train": mwz_train_response_prompt,
  "taxi": mwz_taxi_response_prompt,
  "hospital": mwz_hospital_response_prompt,
  "bus": mwz_bus_response_prompt
}