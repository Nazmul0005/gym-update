import logging
import uuid

from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START

from com.mhire.app.config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AICoach:
    def __init__(self):
        try:
            config = Config()
            self.model = ChatOpenAI(
                openai_api_key=config.openai_api_key,
                model=config.model_name2,
                temperature=1
            )
            # Create a workflow using LangGraph
            workflow = StateGraph(state_schema=MessagesState)
            
            # Define the model node function
            def call_model(state):
                return {"messages": self.model.invoke(state["messages"])}
            
            # Add the model node and connect it
            workflow.add_node("model", call_model)
            workflow.add_edge(START, "model")
            
            # Set up memory persistence
            self.memory = MemorySaver()
            self.app = workflow.compile(checkpointer=self.memory)
            
        except Exception as e:
            logger.error(f"Error initializing AICoach: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize AI Coach: {str(e)}")

    async def chat(self, user_message: str) -> str:
        try:
            # Define the base system prompt for a friendly AI gym coach
            system_prompt = """You are a friendly and supportive AI gym coach named Coach AI. Your role is to:

                1. Provide helpful fitness and nutrition advice in a conversational, friendly manner
                2. Give concise responses for casual conversation and ask follow-up questions when needed to gather information
                3. Collect only ESSENTIAL minimum information before providing detailed solutions, then refine based on user feedback
                4. Provide comprehensive solutions with basic information, then ask if user wants modifications
                5. Stay focused on fitness, health, nutrition, and wellness topics only
                6. Politely redirect off-topic questions back to health and fitness
                7. Be empathetic and understanding while helping users achieve their fitness goals
                8. Give scientifically-backed recommendations in an easy-to-understand way

                Communication guidelines:
                - Never use emojis in your responses
                - Keep responses short and conversational for general chat
                - Gather MINIMAL essential information before providing solutions (for weight loss: just height, weight, target - then provide solution)
                - After giving initial solution, ask: "How does this look? Any specific preferences or issues I should adjust for?"
                - Let users refine the solution by mentioning specific needs rather than asking multiple screening questions upfront
                - For weight loss: collect height, weight, target weight only - then give comprehensive plan
                - For workout plans: collect fitness level and goals only - then provide plan
                - Redirect non-health topics politely: "I'm here to help with fitness and health questions. Let's focus on your wellness goals."

                Approach: Give solutions quickly with minimal info, then refine based on user feedback rather than extensive upfront questioning.

                Always maintain a friendly, conversational tone while being helpful and professional."""
            # Create a unique thread ID for this conversation if not already stored
            if not hasattr(self, 'thread_id'):
                self.thread_id = str(uuid.uuid4())
            
            # Set up configuration with thread ID
            config = {"configurable": {"thread_id": self.thread_id}}
            
            # Create a human message from the user input
            human_message = HumanMessage(content=user_message)
            
            # Get the current state from memory or create a new one with system prompt
            try:
                # Try to get existing conversation
                current_state = self.app.get_state(config)
                # Add the new message to existing messages
                messages = current_state["messages"] + [human_message]
            except Exception:
                # If no existing conversation, start a new one with system prompt
                system_message = SystemMessage(content=system_prompt)
                messages = [system_message, human_message]
            
            # Run the model with the messages
            result = self.app.invoke({"messages": messages}, config)
            
            # Extract the assistant's response
            assistant_message = result["messages"][-1]
            response_content = assistant_message.content
            
            return response_content

        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get AI response: {str(e)}")