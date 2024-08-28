from django.db import models

# Create your models here.

class heartModel(models.Model):

    age=models.IntegerField()
    cp=models.FloatField()
    trestbps=models.FloatField()
    chol=models.FloatField()
    fbs=models.FloatField()
    restecg=models.FloatField()
