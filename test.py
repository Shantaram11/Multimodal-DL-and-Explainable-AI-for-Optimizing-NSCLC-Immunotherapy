import sys, torch, pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow
from qt_material import apply_stylesheet

from UI import Ui_MainWindow                    # your .ui → .py file
from improved import explain_local, ask_local, feedback_local

# Dummy MLP (unchanged)
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size, hidden=[16,8,4], drop=0.3):
        super().__init__()
        seq=[]
        in_dim=input_size
        for h in hidden:
            seq += [nn.Linear(in_dim,h), nn.ReLU(), nn.Dropout(drop)]
            in_dim=h
        seq.append(nn.Linear(in_dim,1))
        self.model = nn.Sequential(*seq)
    def forward(self,x): return torch.sigmoid(self.model(x))

class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow(); self.ui.setupUi(self)

        # hide follow‑up widgets initially
        self.ui.plainTextEdit_feedback.hide()
        self.ui.label_feedback.hide()
        self.ui.pushButton_submit.hide()

        # load PyTorch models
        self.model_bs = torch.load("model_bs.pth", weights_only=False)
        self.model_ps = torch.load("model_ps.pth", weights_only=False)
        self.model_po = torch.load("model_po.pth", weights_only=False)

        # SHAP top features
        self.shap_bs = pd.read_csv("shap_bs.csv")["Features"].values
        self.shap_ps = pd.read_csv("shap_ps.csv")["Features"].values
        self.shap_po = pd.read_csv("shap_po.csv")["Features"].values

        # dataset for predictions
        self.dataset = pd.read_csv("final_merged_rfe.csv").drop(
            columns=["patient_id","Best response","Potential status","Progression occurrence"]
        )

        # button hooks
        self.ui.pushButton_Run.clicked.connect(self.on_run)
        self.ui.pushButton_Reset.clicked.connect(self.on_reset)
        self.ui.pushButton_submit.clicked.connect(self.on_submit)

    # ── UI slots ────────────────────────────────────────────────────
    def on_reset(self):
        self.ui.textBrowser_prediction.clear()
        self.ui.textBrowser_LLMresult.clear()
        self.ui.plainTextEdit_feedback.clear()
        self.ui.plainTextEdit_feedback.hide()
        self.ui.label_feedback.hide()
        self.ui.pushButton_submit.hide()
        self.ui.radioButton_ps.setChecked(True)
        self.ui.spinBox.setValue(0)

    def on_run(self):
        try:
            pid = self.ui.spinBox.value()
            if not (0 <= pid < len(self.dataset)):
                self.ui.textBrowser_LLMresult.setText("Patient id out of range (0‑72).")
                return

            # run prediction
            features_vec = torch.tensor(self.dataset.iloc[pid], dtype=torch.float32)
            if self.ui.radioButton_bs.isChecked():
                outcome = round(self.model_bs(features_vec).item())
                pred_txt = "Progression/Stable Disease" if outcome==0 else "Partial/Complete Response"
                shap_feats = self.shap_bs
            elif self.ui.radioButton_po.isChecked():
                outcome = round(self.model_po(features_vec).item())
                pred_txt = ("First Progression occurs less than 6 months" if outcome==1 else "First Progression occurs more than 6 months")
                shap_feats = self.shap_po
            else:  # potential status
                outcome = round(self.model_ps(features_vec).item())
                pred_txt = "Die within 1 year" if outcome==1 else "Alive more than 1 year"
                shap_feats = self.shap_ps

            self.ui.textBrowser_prediction.setText(pred_txt)

            # call local LLM pipeline
            self.eid, expl = explain_local({"genes": list(shap_feats)})
            self.ui.textBrowser_LLMresult.setText(expl)

            # show follow‑up widgets
            self.ui.plainTextEdit_feedback.show()
            self.ui.label_feedback.show()
            self.ui.pushButton_submit.show()
        except Exception as e:
            print(e)

    def on_submit(self):
        question = self.ui.plainTextEdit_feedback.toPlainText().strip()
        if not question: return
        answer = ask_local(question, self.eid)
        self.ui.textBrowser_LLMresult.setText(answer)
        self.ui.plainTextEdit_feedback.clear()
        # optional: treat each question as implicit rating 5
        feedback_local(self.eid, 5, f"Asked: {question}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_pink.xml')
    win = UI(); win.show()
    sys.exit(app.exec_())