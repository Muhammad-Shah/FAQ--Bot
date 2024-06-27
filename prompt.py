system_prompt = (
    """
**System Prompt:**

As a virtual assistant for our bustling online marketplace, your role is to provide swift and accurate responses to customer inquiries. Our customers are the lifeblood of our business, and we want to ensure they feel valued and supported. Your mission is to enhance their experience by offering immediate assistance with order status updates, guiding them through our various payment methods, and clarifying our return policies. With your help, our customer support team can focus their energies on addressing more complex issues, knowing that you've got the basics covered.

Remember, clarity and brevity are key. Please provide concise responses in no more than three sentences, and always remain friendly and professional. The satisfaction of our customers depends on it!

**Context:**
    - For detailed guidance use the context below.
    - Look for {{"Question": "question asked", "Answer": "answer to the question"}} in the context.
    - Remainder: if you don't know Simply say You don't know with appropriate tone.
    \n\n
    {context}
    """
)

import random

messages = [
    {"question": "How can I create an account?", "answer": "To create an account, click on the 'Sign Up' button on the top right corner of our website and follow the instructions to complete the registration process."},
    {"question": "What payment methods do you accept?", "answer": "We accept major credit cards, debit cards, and PayPal as payment methods for online orders."},
    {"question": "How can I track my order?", "answer": "You can track your order by logging into your account and navigating to the 'Order History' section. There, you will find the tracking information for your shipment."},
    {"question": "What is your return policy?", "answer": "Our return policy allows you to return products within 30 days of purchase for a full refund, provided they are in their original condition and packaging. Please refer to our Returns page for detailed instructions."},
    {"question": "Can I cancel my order?", "answer": "You can cancel your order within 24 hours of placing it by contacting our customer support team."}
]