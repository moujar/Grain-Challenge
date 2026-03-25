# Submission Guidelines

## Starter Kit

Download the **starter kit** from the **Files** section of the competition page. It contains everything you need to get started, including sample code and documentation.

---

## Pretrained Model

Submit your **pretrained model** as shown in the provided Colab notebook. Only **inference** will be run on CodaBench — your model will not be retrained during evaluation, so ensure your weights are saved and loaded correctly as demonstrated in the Colab.

---

## Training Code (Optional)

You may include an additional file (any name **other than** `model.py`) that contains your training code or documents your training process. This allows us to review how the model was trained. This file will not be executed during evaluation.

---

## Submission Format

Open a terminal and navigate to the directory that contains model.py.

Run the following command:

```
zip submission.zip model.py
```

If you are including a training code file, add it to the archive as well (replace `training.py` with your actual filename):

```
zip submission.zip model.py training.py
```

This will create a file named submission.zip containing your files at the root of the archive. Upload the generated submission.zip as your submission.