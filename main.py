import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import requests

def init_azure_services():
    load_dotenv()
    azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
    azure_oai_key = os.getenv("AZURE_OAI_KEY")
    azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")
    azure_search_index = os.getenv("AZURE_SEARCH_INDEX")

    client = AzureOpenAI(
        api_key=azure_oai_key,
        api_version="2023-12-01-preview",
        azure_endpoint=azure_oai_endpoint
    )
    search_config = {
        "endpoint": azure_search_endpoint,
        "key": azure_search_key,
        "index_name": azure_search_index
    }
    return client, azure_oai_deployment, search_config

def search_documents(search_config, question, top_k=3):
    url = f"{search_config['endpoint']}/indexes/{search_config['index_name']}/docs/search?api-version=2021-04-30-Preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": search_config['key']
    }
    data = {
        "search": question,
        "top": top_k,
        "queryType": "simple",
        "select": "*"
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json()
        docs = results.get("value", [])
        context = "\n\n".join([doc.get("content", "") for doc in docs])
        return context.strip()
    except Exception as e:
        print(f"Erreur Azure Search : {e}")
        return ""

def ask_gpt(client, deployment_name, question, context=None, strict_rag=False):
    if context and strict_rag:
        system_prompt = (
            "Vous êtes un assistant qui répond uniquement à partir du contexte fourni ci-dessous. "
            "Si la réponse n'est pas explicitement présente dans le contexte, répondez strictement : "
            "\"Aucune information trouvée dans la base documentaire.\" "
            "N'utilisez aucune connaissance externe."
        )
        prompt = f"Contexte :\n{context}\n\nQuestion : {question}"
    else:
        system_prompt = "Vous répondez précisément à la question."
        prompt = question

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            temperature=0.1,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erreur GPT : {e}"

def boucle_questions(client, deployment_name, search_config):
    while True:
        print("Voulez-vous utiliser le RAG (contexte documentaire) ? (o/n)")
        use_rag = input("> ").strip().lower() == "o"
        print("Posez votre question :")
        question = input("> ").strip()
        context = ""
        strict_rag = False
        if use_rag:
            context = search_documents(search_config, question)
            strict_rag = True
            if not context:
                print("Aucun contexte trouvé, question envoyée sans contexte.")
                strict_rag = False
        print("\n===== Réponse Azure OpenAI =====")
        print(ask_gpt(client, deployment_name, question, context if use_rag and context else None, strict_rag))
        print("\nVoulez-vous poser une autre question ? (o/n)")
        recommencer = input("> ").strip().lower()
        if recommencer != "o":
            print("Au revoir !")
            break

if __name__ == "__main__":
    client, deployment_name, search_config = init_azure_services()
    boucle_questions(client, deployment_name, search_config)

