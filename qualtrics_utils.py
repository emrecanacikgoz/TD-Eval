survey_header = """
[[AdvancedFormat]]
"""

# TODO: add definitions of these dimensions here
instr_prefix = """
[[Block:Instructions]]
[[Question:Text]]
Thank you for participating in this human evaluation study. We have provided 20 dialogues where an AI agent is responding to a user request, and we want you to evaluate the responses. This survey should take roughly 60-75 minutes. Progress will be saved automatically, so you can complete it in multiple sessions, but <strong>make sure you are using the same browser each time.</strong> This reference page will be available as a separate PDF file as well for reference. Please rate the following dialogue responses of AI agents from 1 (worst) to 10 (best) on the following qualities: Conversation Consistency, Backend Knowledge Consistency, and Policy Completeness. The metric scales are listed below. <br/><br/>

<strong>Conversation Consistency Assessment (1-5 Scale)</strong><br/></br>

<strong>Very Good (5)</strong>: Response is completely consistent with the previous conversation context, with no inconsistencies or errors.
<strong>Good (4)</strong>: Response is mostly consistent with the context. Only minor improvements are needed.
<strong>Fair (3)</strong>: Response is somewhat consistent but contains noticeable inconsistencies or lacks depth in addressing the context.
<strong>Bad (2)</strong>: Response shows limited consistency with the conversation context and requires significant improvement.
<strong>Very Bad (1)</strong>: Response is incoherent or completely inconsistent with the conversation context.

<strong>Backend Knowledge Consistency Assessment (1-5 Scale)</strong><br/><br/>

<strong>Very Good (5)</strong>: Response is completely consistent with the database results, with no inconsistencies or errors.
<strong>Good (4)</strong>: Response is mostly consistent with the database results. Only minor improvements are needed.
<strong>Fair (3)</strong>: Response is sufficiently consistent with the database results but contains noticeable inconsistencies or lacks depth in addressing the results.
<strong>Bad (2)</strong>: Response shows limited consistency with the database results and requires significant improvement.
<strong>Very Bad (1)</strong>: Response is incoherent or completely inconsistent with the database results.

<strong>Policy Completeness Assessment (1-5 Scale)</strong><br/><br/>

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
1 (Worst)
[[Answer]]
2
[[Answer]]
3
[[Answer]]
4
[[Answer]]
5 (Best)
"""