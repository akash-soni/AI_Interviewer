from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
import os

load_dotenv(override=True)

model_client = OpenAIChatCompletionClient( model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")
                                          )


# defining our Agent

#1. Interviewer Agent
#2. Interviewee Agent
#3. Career coach Agent

job_position = "Software Engineer"

interviewer = AssistantAgent(
    name="Interviewer", 
    description=f"An expert interviewer for a {job_position} position.",
    model_client=model_client,
    system_message='''You are an professional interviewer for a {job_position} position. 
    Ask one clear question at a time, and wait for the candidate to respond before asking the next question.
    Your job is to continue and ask questions until, dont pay any attention to the responses from career coach.
    Ask one quesion at a time, in total coveiring technical skills, problem-solving abilities, and cultural fit.
    After asking 3 questions, say 'TERMINATE' to end the interview.'''
    )



interviewee = UserProxyAgent(
    name = "Interviewee",
    description=f"A candidate applying for a {job_position} position.",
    input_func=input
)



# 3. Career Coach Agent
career_coach = AssistantAgent(
    name="Career_Coach",
    description=f''' A career coach who provides feedback and advice to the interviewee for a {job_position} position.''',
    model_client=model_client,
    system_message=f'''You are a career coach. 
    After the interview, provide constructive feedback to the interviewee for a {job_position} on their performance, including strengths and areas for improvement.'''
)

terminate_condition = TextMentionTermination(
    text="TERMINATE")

team = RoundRobinGroupChat(
    participants=[interviewer, interviewee, career_coach],
    termination_condition=terminate_condition,
        max_turns=20
    )



# Running the interview
stream = team.run_stream(task="Conducting an interview for a {job_position}",)

async def main():
    await Console(stream)



if(__name__ == "__main__"):
    import asyncio
    asyncio.run(main())


