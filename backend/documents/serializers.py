from rest_framework import serializers
from .models import Document


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'file', 'uploaded_at', 'processed']
        read_only_fields = ['uploaded_at', 'processed']