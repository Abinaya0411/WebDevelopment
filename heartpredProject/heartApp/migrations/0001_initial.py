# Generated by Django 3.0.2 on 2024-08-26 02:16

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='heartModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('age', models.IntegerField()),
                ('cp', models.FloatField()),
                ('trestbps', models.FloatField()),
                ('chol', models.FloatField()),
                ('fbs', models.FloatField()),
                ('restecg', models.FloatField()),
            ],
        ),
    ]
