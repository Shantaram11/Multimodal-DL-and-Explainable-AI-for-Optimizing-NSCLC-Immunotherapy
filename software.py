import sys
import torch
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow
from UI import Ui_MainWindow
import pickle
# from model import MLP
from qt_material import apply_stylesheet
import openai
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims=[16, 8, 4], dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))
class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()  # create an instance of the UI class
        self.ui.setupUi(self)
        self.ui.plainTextEdit_feedback.hide()
        self.ui.label_feedback.hide()
        self.ui.pushButton_submit.hide()
        self.model_bs = torch.load("model_bs.pth", weights_only=False)
        self.model_ps = torch.load("model_ps.pth", weights_only=False)
        self.model_po = torch.load("model_po.pth", weights_only=False)
        self.shap_bs_features = pd.read_csv("shap_bs.csv")["Features"].values
        self.shap_ps_features = pd.read_csv("shap_ps.csv")["Features"].values
        self.shap_po_features = pd.read_csv("shap_po.csv")["Features"].values
        self.ui.pushButton_Run.clicked.connect(self.on_run_clicked_run)
        self.dataset = pd.read_csv("final_merged_rfe.csv").drop(columns=["patient_id", "Best response", "Potential status", "Progression occurrence"])
        self.ui.pushButton_Reset.clicked.connect(self.on_run_clicked_reset)
        self.ui.pushButton_submit.clicked.connect(self.on_run_clicked_submit)
        self.client = openai.OpenAI(api_key="")  # or use api_key="your-key"

    def on_run_clicked_submit(self):
        questions = str(self.ui.plainTextEdit_feedback.toPlainText())
        reply = self.get_LLM_further_questions(questions)
        self.ui.textBrowser_LLMresult.setText(reply)
        self.ui.plainTextEdit_feedback.setPlainText("")
    def on_run_clicked_reset(self):
        self.ui.textBrowser_prediction.setText("")
        self.ui.textBrowser_LLMresult.setText("")
        self.ui.radioButton_ps.click()
        self.ui.spinBox.setValue(0)
        self.ui.plainTextEdit_feedback.hide()
        self.ui.label_feedback.hide()
        self.ui.pushButton_submit.hide()
    def on_run_clicked_run(self):
        id = self.ui.spinBox.value()
        if int(id) > 72 or int(id) < 0:
            self.ui.textBrowser_LLMresult.setText(f"Only 73 patients existed. Patient id {id} is not existed. Valid patient id range is [0, 72]")
        else:
            data = torch.tensor(self.get_data(id), dtype=torch.float32)
            if self.ui.radioButton_bs.isChecked():
                self.prediction = "Best response"
                self.prediction_bs = round(self.model_bs(data).tolist()[0], 0)
                self.prediction_bs_word = "Progression/Stable Disease" if self.prediction_bs == 0.0 else "Partial/Complete Response"
                self.ui.textBrowser_prediction.setText(self.prediction_bs_word)
                llm_results = self.get_LLM(self.shap_bs_features)
            elif self.ui.radioButton_ps.isChecked():
                self.prediction = "Potential status"
                self.prediction_ps = round(self.model_ps(data).tolist()[0], 0)
                self.prediction_ps_word = "Die within 1 year" if self.prediction_ps == 1.0 else "alive more than 1 year"
                self.ui.textBrowser_prediction.setText(self.prediction_ps_word)
                llm_results = self.get_LLM(self.shap_ps_features)
            elif self.ui.radioButton_po.isChecked():
                self.prediction = "Progression occurrence"
                self.prediction_po = round(self.model_po(data).tolist()[0], 0)
                self.prediction_po_word = "First progression occurs within 6 months" if self.prediction_po == 1.0 else "First progression occurs beyond 6 months"
                self.ui.textBrowser_prediction.setText(self.prediction_po_word)
                llm_results = self.get_LLM(self.shap_po_features)
            self.ui.textBrowser_LLMresult.setText(str(llm_results))
            self.ui.plainTextEdit_feedback.show()
            self.ui.label_feedback.show()
            self.ui.pushButton_submit.show()


    def get_data(self, id):
        return self.dataset.iloc[id]
    def get_LLM(self, features):
        self.message = [
                {"role": "system", "content": f"""\
                You are an expert biomedical researcher specializing in non-small cell lung cancer (NSCLC) and in-depth analysis of scientific literature.
                You are provided with the following features for deep analysis:{str(features)}
            """},
                {"role": "user", "content": """These features represent the most predictive SHAP values identified by a machine learning model trained to predict patient outcomes following immunotherapy for NSCLC.


                    The model predicts the following outcomes:
                    - Best response: Best observed RECIST [1] response (i.e., progression, stable disease, partial
                      response, or complete response) during the patient's follow-up after first-line immunotherapy
                      initiation.
                    - OS:  Vital status: Duration from the initiation of first-line immunotherapy (with or without
                      chemotherapy) to the patient's death or last available status update. 
                    - PFS:  Progression: Duration from the initiation of first-line immunotherapy to the occurrence
                      of the first progression event or last available status update, including the emergence of new
                      lesions or the progression of pre-existing ones.

                    Please search and synthesize the most recent, high-quality, peer-reviewed literature — from sources such as PubMed, NEJM, Lancet Oncology, ClinicalTrials.gov, Nature Medicine, and The Cancer Genome Atlas (TCGA) — to provide a detailed explanation of how these features contribute to NSCLC outcomes following immunotherapy.

                    Your response should include:
                    - A summary of key findings
                    - A detailed explanation of relevant biological pathways
                    - For each feature:
                      - Its prognostic or predictive role (e.g., risk or protective factor)
                      - Its mechanism of action, when known
                      - The level of evidence (e.g., preclinical, retrospective, prospective, RCT, meta-analysis)
                    - Prioritize studies from 2020–2024, but include landmark earlier findings when appropriate
                    - give IEEE format citations

                    Present your findings clearly and relatively concisely, using accessible language suitable for a mixed audience of clinicians and patients. Avoid overwhelming technical jargon. Use headings and bullet points for readability.
            """}
            ]
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages = self.message
        )
        reply = response.choices[0].message.content
        self.message.append({"role": "assistant", "content": reply})

        return reply

    def get_LLM_further_questions(self, questions):
        self.message.append({"role": "user", "content": questions})
        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages = self.message
        )
        reply = response.choices[0].message.content
        self.message.append({"role": "assistant", "content": reply})
        return reply







if __name__ == "__main__":
    # glossary = pd.read_csv("final_glossary.csv")
    # description = get_description(glossary, ["patient_id", "Progression occurrence"])
    # print(description)
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_pink.xml')
    window = UI()
    window.show()
    sys.exit(app.exec_())
