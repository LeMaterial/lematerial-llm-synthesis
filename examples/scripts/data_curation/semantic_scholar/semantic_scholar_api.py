import requests


class SemanticScholarAPI:
    def __init__(self):
        self.base_url = (
            "https://api.semanticscholar.org/graph/v1/paper/search?query="
        )
        self.fields = "&fields=url,abstract,authors"

    def get_response(self, query):
        query_request = self.base_url + query + self.fields + "&limit=3"
        response = requests.get(query_request)
        """soup = BeautifulSoup(response.text, "html.parser")
        markdown = html2text.html2text(str(soup))"""
        print(response)
        print(response.json())
        return


SemanticScholarAPI().get_response("materials synthesis")
