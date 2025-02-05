survey_header = """
[[AdvancedFormat]]
"""

instr_prefix = """
[[Block:Instructions]]
[[Question:Text]]
Thank you for participating in this annotation task. We have provided 20 dialogues where an AI agent is responding to a user request, and we want you to evaluate the responses. This survey should take roughly 60-75 minutes. Progress will be saved automatically, so you can complete it in multiple sessions, but <strong>make sure you are using the same browser each time.</strong> Please rate individual dialogue responses of AI agents from 1 (worst) to 5 (best) on the following qualities: Conversation Consistency, Backend Knowledge Consistency, and Policy Completeness, the metric definitions are below. In addition, please rate the full dialogue in terms of task completion and response coherence. <br/><br/>

<strong>Conversation Consistency</strong><br/>
How much an agent's response align with the context of conversation context. </br>
<ul>
<li><strong>Relevance</strong>: The response directly relates to the dialogue history and the current user query.</li>
<li> <strong>Topic Consistency</strong>: The response remains on-topic with the dialogue history and the user query.</li>
<li><strong>Coherence</strong>: The response logically continues the progression of the dialogue.</li>
</ul>
</br>

<strong>Scoring Scale</strong><br/>
<ol>
<li><strong>Very Good (5)</strong>: Response is completely consistent with the previous conversation context, with no inconsistencies or errors.</li>
<li><strong>Good (4)</strong>: Response is mostly consistent with the context. Only minor improvements are needed.</li>
<li><strong>Fair (3)</strong>: Response is somewhat consistent but contains noticeable inconsistencies or lacks depth in addressing the context.</li>
<li><strong>Bad (2)</strong>: Response shows limited consistency with the conversation context and requires significant improvement.</li>
<li><strong>Very Bad (1)</strong>: Response is incoherent or completely inconsistent with the conversation context.</li>
</ol>
</br></br>

<strong>Backend Knowledge Consistency</strong><br/>
How well an agent's response aligns with information provided by backend database results.</br>
<ul>
<li><strong>Accuracy</strong>: The response directly reflects the information in the database results.</li>
<li> <strong>Topic Consistency</strong>: The response stays on-topic with the database results and the dialogue context.</li>
<li><strong>Coherence</strong>: The response logically incorporates and progresses based on the database results.</li>
</ul>
</br></br>

<strong>Scoring Scale</strong><br/>
<ol>
<li><strong>Very Good (5)</strong>: Response is completely consistent with the database results, with no inconsistencies or errors.</li>
<li><strong>Good (4)</strong>: Response is mostly consistent with the database results. Only minor improvements are needed.</li>
<li><strong>Fair (3)</strong>: Response is sufficiently consistent with the database results but contains noticeable inconsistencies or lacks depth in addressing the results.</li>
<li><strong>Bad (2)</strong>: Response shows limited consistency with the database results and requires significant improvement.</li>
<li><strong>Very Bad (1)</strong>: Response is incoherent or completely inconsistent with the database results.</li>
</ol>
</br></br>

<strong>Policy Compliance</strong><br/>
How well an agent's response adheres to the expected policy protocol. </br>
<ul>
<li><strong>Number of Suggestions</strong>: Providing suggestions only when the database results are small enough to do so.</li>
<li> <strong>Information Gathering</strong>: Requesting required, relevant information (slots) from the user before offering suggestions or booking services.</li>
<li><strong>Appropriate Timing</strong>: Avoiding premature actions, such as making a booking or suggesting a service too early in the conversation.</li>
<li><strong>Alignment with Policy</strong>: Avoiding actions that do not align with the suggested flow of interaction in the policy, when available.</li>

</ul>
</br>

<strong>Expected Policy</strong></br>
The chatbot response should depend on the database results and dialogue history: <br/>
<ul>
<li>If the database results return a number larger than 10: Indicate the number of entries that match the user's query and request additional information if needed to narrow down the results.</li>
<li>If the database results return values less than 10: If vital details are missing, request additional information. Otherwise, provide the relevant entries to the user</li>
</ul>
<br/><br/>

<strong>Scoring Scale</strong><br/>
<ol>
<li><strong>Very Good</strong>: Response fully follows policy protocol with no errors or omissions.</li>
<li><strong>Good</strong>: Response mostly follows policy protocol, with only minor room for improvement.</li>
<li><strong>Fair</strong>: Response sufficiently follows policy protocol but has clear areas where it could improve in completeness or timing.</li>
<li><strong>Bad</strong>: Response does not adequately follow policy protocol, though there may be partial adherence.</li>
<li><strong>Very Bad</strong>: Response fails to follow policy protocol and is incomplete or incoherent.</li>
</ol>
"""

dial_block_prefix = """
[[Block:Dialogue #{index}]]
"""

matrix_q_format = """
[[Question:Matrix]]
Read the dialogue context below:<br/><br/>
<hr>
{dial_hist}
<hr>
{db_results}
<hr>
<strong>Agent:</strong> {agent_resp}<br/>
<hr>
<br/><br/>

Rate the Conversation Consistency, Backend Knowledge Consistency, and Policy Completeness from 1 to 5 for the above response.<br/><br/>

[[Choices]]
Conversation Consistency
Backend Knowledge Consistency
Policy Completeness

[[AdvancedAnswers]]
[[Answer]]
Very Bad
[[Answer]]
Bad
[[Answer]]
Fair
[[Answer]]
Good
[[Answer]]
Very Good
"""

dial_completion_rate_q = """
[[Question:MC:SingleAnswer:Horizontal]]
Read the dialogue context below:<br/><br/>
<hr>
{dial_hist}<br/>
<hr>
<br/>
Task Completion: Rate your satisfaction with the chatbot's answers to the user's questions. How well would you rate the bot's ability to <strong>fully answer the users' queries?</strong> <br/><br/>

[[Choices]]
Very Bad
Bad
Fair
Good
Very Good
"""

dial_satisfaction_rate_q = """
[[Question:MC:SingleAnswer:Horizontal]]
Read the dialogue context below:<br/><br/>
<hr>
{dial_hist}
<hr><br/>
Response Coherence: Rate your satisfaction with the chatbot's answers to the user's questions. How would you rate your satisfaction with bot's responses in terms of being <strong>easy to understand and helpful?</strong> <br/><br/>

[[Choices]]
Very Bad
Bad
Fair
Good
Very Good
"""