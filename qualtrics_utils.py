survey_header = """
[[AdvancedFormat]]
"""

instr_prefix = """
[[Block:Instructions]]
[[Question:Text]]
Thank you for participating in this human evaluation study. We have provided 20 dialogues where an AI agent is responding to a user request, and we want you to evaluate the responses. This survey should take roughly 60-75 minutes. Progress will be saved automatically, so you can complete it in multiple sessions, but <strong>make sure you are using the same browser each time.</strong> This reference page will be available as a separate PDF file as well for reference. Please rate the following dialogue responses of AI agents from 1 (worst) to 10 (best) on the following qualities: Conversation Consistency, Backend Knowledge Consistency, and Policy Completeness. The metric scales are listed below. <br/><br/>

<strong>Conversation Consistency Assessment (1-5 Scale)</strong><br/>
How an agent's response align with the context of conversation context. </br>
<ul>
<li><strong>Relevance</strong>: The response directly relates to the dialogue history and the current user query.</li>
<li> <strong>Topic Consistency</strong>: The response remains on-topic with the dialogue history and the user query.</li>
<li><strong>Coherence</strong>: The response logically continues the progression of the dialogue.</li>
</ul>
</br></br>

<strong>Very Good (5)</strong>: Response is completely consistent with the previous conversation context, with no inconsistencies or errors.
<strong>Good (4)</strong>: Response is mostly consistent with the context. Only minor improvements are needed.
<strong>Fair (3)</strong>: Response is somewhat consistent but contains noticeable inconsistencies or lacks depth in addressing the context.
<strong>Bad (2)</strong>: Response shows limited consistency with the conversation context and requires significant improvement.
<strong>Very Bad (1)</strong>: Response is incoherent or completely inconsistent with the conversation context.

<strong>Backend Knowledge Consistency Assessment (1-5 Scale)</strong><br/>
How well an agent's response aligns with information provided in the policy or database results. </br>
<ul>
<li><strong>Relevance</strong>: The response directly reflects the information in the database results.</li>
<li> <strong>Topic Consistency</strong>: The response stays on-topic with the database results and the dialogue context.</li>
<li><strong>Coherence</strong>: The response logically incorporates and progresses based on the database results.</li>
</ul>
</br></br>

<strong>Very Good (5)</strong>: Response is completely consistent with the database results, with no inconsistencies or errors.
<strong>Good (4)</strong>: Response is mostly consistent with the database results. Only minor improvements are needed.
<strong>Fair (3)</strong>: Response is sufficiently consistent with the database results but contains noticeable inconsistencies or lacks depth in addressing the results.
<strong>Bad (2)</strong>: Response shows limited consistency with the database results and requires significant improvement.
<strong>Very Bad (1)</strong>: Response is incoherent or completely inconsistent with the database results.

<strong>Policy Completeness Assessment (1-5 Scale)</strong><br/>
How well an agent's response adheres to the expected policy protocol. </br>
<ul>
<li><strong>Relevance of Suggestions</strong>: Providing suggestions only when the database results are small enough to do so.</li>
<li> <strong>Information Gathering</strong>: Requesting required, relevant information (slots) from the user before offering suggestions or booking services.</li>
<li><strong>Appropriate Timing</strong>: Avoiding premature actions, such as making a booking or suggesting a service too early in the conversation.</li>
</ul>
</br></br>

<strong>Expected Policy</strong></br>
The chatbot response should depend on the database results and dialogue history: <br/>
<ol>
<li>If the database results return a number: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results.</li>
<li>If the database results return values: If vital details are missing based on the dialogue history, request additional information. Otherwise, provide the relevant entries to the user</li>
</ol>
<br/><br/>

<strong>Very Good (5)</strong>: Response fully follows policy protocol with no errors or omissions.
<strong>Good (4)</strong>: Response mostly follows policy protocol, with only minor room for improvement.
<strong>Fair (3)</strong>: Response sufficiently follows policy protocol but has clear areas where it could improve in completeness or timing.
<strong>Bad (2)</strong>: Response does not adequately follow policy protocol, though there may be partial adherence.
<strong>Very Bad (1)</strong>: Response fails to follow policy protocol and is incomplete or incoherent.
"""

dial_block_prefix = """
[[Block:Dialogue #{index}]]
"""

matrix_q_format = """
[[Question:Matrix]]
Read the dialogue context below:<br/><br/>
---<br/>
{dial_hist}
---<br/>
{db_results}
---<br/>
<strong>Agent Response:</strong> {agent_resp}<br/><br/>

Rate the Conversation Consistency, Backend Knowledge Consistency, and Policy Completeness from 1 to 5 for the above response.<br/><br/>

[[Choices]]
Conversation Consistency
Backend Knowledge Consistency
Policy Completeness

[[AdvancedAnswers]]
[[Answer]]
1 (Very Bad)
[[Answer]]
2 (Bad)
[[Answer]]
3 (Fair)
[[Answer]]
4 (Good)
[[Answer]]
5 (Very Good)
"""

dial_completion_rate_q = """
[[Question:MC:SingleAnswer:Horizontal]]
Read the dialogue context below:<br/><br/>
---<br/>
{dial_hist}
---<br/>
Rate your <strong>satisfaction</strong> with the chatbot's answers to the user's questions. How well would you rate the bot's ability to <strong>completely</strong> answer the users' queries? <br/><br/>

[[Choices]]
1 (Very Bad)
2 (Bad)
3 (Fair)
4 (Good)
5 (Very Good)
"""

dial_satisfaction_rate_q = """
[[Question:MC:SingleAnswer:Horizontal]]
Read the dialogue context below:<br/><br/>
---<br/>
{dial_hist}
---<br/>
Rate your <strong>satisfaction</strong> with the chatbot's answers to the user's questions. How would you rate your satisfaction with bot's responses in terms of being easy to understand and helpful? <br/><br/>

[[Choices]]
1 (Very Bad)
2 (Bad)
3 (Fair)
4 (Good)
5 (Very Good)
"""