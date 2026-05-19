from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity
from graph.graph import process_query


class LabChatBot(ActivityHandler):

    async def on_message_activity(self, turn_context: TurnContext):
        user_message = turn_context.activity.text.strip()

        await turn_context.send_activity("Recherche en cours dans la base de connaissances...")

        try:
            result = process_query(user_message)

            answer = result.get("answer", "Information non disponible.")
            sources = result.get("sources", [])
            query_type = result.get("query_type", "unknown")
            relevance = result.get("relevance_score", 0)
            hallucination = result.get("hallucination_score", 0)

            if query_type == "out_of_scope":
                response = "Cette question est hors du perimetre de la base de connaissances du laboratoire."
            else:
                response = f"**Reponse**\n\n{answer}"
                if sources:
                    sources_text = "\n".join([f"- {s}" for s in sources[:5]])
                    response += f"\n\n**Sources**\n{sources_text}"
                response += f"\n\nPertinence : {relevance:.2f} | Fiabilite : {hallucination:.2f}"

        except Exception as e:
            response = f"Une erreur est survenue : {str(e)}"

        await turn_context.send_activity(response)

    async def on_members_added_activity(self, members_added, turn_context: TurnContext):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(
                    "Bonjour ! Je suis le chatbot du laboratoire.\n"
                    "Posez-moi vos questions sur les composes testes, "
                    "les resultats d'etudes ou les donnees de R&D."
                )
