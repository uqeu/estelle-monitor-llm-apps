import streamlit as st
from support_runtime import CustomerSupportAIAgent
import os

# Set up the Streamlit App
st.title("AI Customer Support Agent with Memory 🛒")
st.caption("Chat with a customer support assistant who remembers your past interactions.")

# Set the OpenAI API key
openai_api_key = st.text_input("Enter OpenAI API Key", type="password")

if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key


    # Initialize the CustomerSupportAIAgent
    support_agent = CustomerSupportAIAgent()

    # Sidebar for customer ID and memory view
    st.sidebar.title("Enter your Customer ID:")
    previous_customer_id = st.session_state.get("previous_customer_id", None)
    customer_id = st.sidebar.text_input("Enter your Customer ID")

    if customer_id != previous_customer_id:
        st.session_state.messages = []
        st.session_state.previous_customer_id = customer_id
        st.session_state.customer_data = None

    # Add button to generate synthetic data
    if st.sidebar.button("Generate Synthetic Data"):
        if customer_id:
            with st.spinner("Generating customer data..."):
                st.session_state.customer_data = support_agent.generate_synthetic_data(customer_id)
            if st.session_state.customer_data:
                st.sidebar.success("Synthetic data generated successfully!")
            else:
                st.sidebar.error("Failed to generate synthetic data.")
        else:
            st.sidebar.error("Please enter a customer ID first.")

    if st.sidebar.button("View Customer Profile"):
        if st.session_state.customer_data:
            st.sidebar.json(st.session_state.customer_data)
        else:
            st.sidebar.info("No customer data generated yet. Click 'Generate Synthetic Data' first.")

    if st.sidebar.button("View Memory Info"):
        if customer_id:
            memories = support_agent.get_memories(user_id=customer_id)
            if memories:
                st.sidebar.write(f"Memory for customer **{customer_id}**:")
                if memories and "results" in memories:
                    for memory in memories["results"]:
                        if "memory" in memory:
                            st.write(f"- {memory['memory']}")
            else:
                st.sidebar.info("No memory found for this customer ID.")
        else:
            st.sidebar.error("Please enter a customer ID to view memory info.")

    # Initialize the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    query = st.chat_input("How can I assist you today?")

    if query and customer_id:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate and display response
        with st.spinner("Generating response..."):
            answer = support_agent.handle_query(query, user_id=customer_id)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    elif not customer_id:
        st.error("Please enter a customer ID to start the chat.")

else:
    st.warning("Please enter your OpenAI API key to use the customer support agent.")
