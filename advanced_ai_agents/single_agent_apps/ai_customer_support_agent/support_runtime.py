"""The support application runtime; usable from Streamlit or a deployment probe."""
import os
import json
from datetime import datetime, timedelta
import streamlit as st
from openai import OpenAI
from mem0 import Memory

class CustomerSupportAIAgent:
    def __init__(self, memory_config=None):
        # Initialize Mem0 with Qdrant as the vector store
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                }
            },
            # mem0 defaults to v1.0, whose search()/get_all() return a bare
            # list; the code below expects the v1.1 {"results": [...]} shape.
            "version": "v1.1",
        }
        try:
            self.memory = Memory.from_config(memory_config if memory_config is not None else config)
        except Exception as e:
            st.error(f"Failed to initialize memory: {e}")
            st.stop()  # Stop execution if memory initialization fails

        self.client = OpenAI(timeout=60, max_retries=0)
        self.app_id = "customer-support"

    def handle_query(self, query, user_id=None):
        try:
            # Search for relevant memories
            relevant_memories = self.memory.search(query=query, user_id=user_id)

            # Build context from relevant memories
            context = "Relevant past information:\n"
            if relevant_memories and "results" in relevant_memories:
                for memory in relevant_memories["results"]:
                    if "memory" in memory:
                        context += f"- {memory['memory']}\n"

            # Generate a response using OpenAI
            full_prompt = f"{context}\nCustomer: {query}\nSupport Agent:"
            response = self.client.chat.completions.create(
                model=os.environ.get("SUPPORT_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are a customer support AI agent for TechGadgets.com, an online electronics store."},
                    {"role": "user", "content": full_prompt}
                ]
            )
            answer = response.choices[0].message.content

            # Add the query and answer to memory
            self.memory.add(query, user_id=user_id, metadata={"app_id": self.app_id, "role": "user"})
            self.memory.add(answer, user_id=user_id, metadata={"app_id": self.app_id, "role": "assistant"})

            return answer
        except Exception as e:
            st.error(f"An error occurred while handling the query: {e}")
            return "Sorry, I encountered an error. Please try again later."

    def get_memories(self, user_id=None):
        try:
            # Retrieve all memories for a user
            return self.memory.get_all(user_id=user_id)
        except Exception as e:
            st.error(f"Failed to retrieve memories: {e}")
            return None

    def generate_synthetic_data(self, user_id: str) -> dict | None:
        try:
            today = datetime.now()
            order_date = (today - timedelta(days=10)).strftime("%B %d, %Y")
            expected_delivery = (today + timedelta(days=2)).strftime("%B %d, %Y")

            prompt = f"""Generate a detailed customer profile and order history for a TechGadgets.com customer with ID {user_id}. Include:
            1. Customer name and basic info
            2. A recent order of a high-end electronic device (placed on {order_date}, to be delivered by {expected_delivery})
            3. Order details (product, price, order number)
            4. Customer's shipping address
            5. 2-3 previous orders from the past year
            6. 2-3 customer service interactions related to these orders
            7. Any preferences or patterns in their shopping behavior

            Format the output as a JSON object."""

            response = self.client.chat.completions.create(
                model=os.environ.get("SUPPORT_MODEL", "gpt-4"),
                messages=[
                    {"role": "system", "content": "You are a data generation AI that creates realistic customer profiles and order histories. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            )

            customer_data = json.loads(response.choices[0].message.content)

            # Add generated data to memory
            for key, value in customer_data.items():
                if isinstance(value, list):
                    for item in value:
                        self.memory.add(
                            json.dumps(item), 
                            user_id=user_id, 
                            metadata={"app_id": self.app_id, "role": "system"}
                        )
                else:
                    self.memory.add(
                        f"{key}: {json.dumps(value)}", 
                        user_id=user_id, 
                        metadata={"app_id": self.app_id, "role": "system"}
                    )

            return customer_data
        except Exception as e:
            st.error(f"Failed to generate synthetic data: {e}")
            return None
