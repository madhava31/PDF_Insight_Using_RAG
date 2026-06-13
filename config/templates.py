"""Jinja2 Template Loader Module

This module provides TemplateLoader for rendering Jinja2 templates from files.
Templates are used to compose system and user prompts with context, history, and questions.
"""
import os
from jinja2 import Environment, FileSystemLoader, TemplateNotFound


class TemplateLoader:
    """Loads and renders Jinja2 templates from the templates/ directory."""

    def __init__(self, template_dir: str = None):
        """
        Initialize the template loader.
        
        Args:
            template_dir: Path to templates directory. Defaults to ./templates/
        """
        if template_dir is None:
            # Resolve templates/ relative to project root (parent of config/)
            config_dir = os.path.dirname(os.path.abspath(__file__))
            template_dir = os.path.join(os.path.dirname(config_dir), "templates")
        
        self.template_dir = template_dir
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def render(self, template_name: str, context: dict) -> str:
        """
        Render a template with the given context.
        
        This method loads a Jinja2 template by name and renders it with the provided variables.
        
        Args:
            template_name: Name of the template file (without .jinja extension)
            context: Dictionary of variables to pass to the template
            
        Returns:
            Rendered template string
            
        Raises:
            TemplateNotFound: If template file does not exist
        """
        try:
            template = self.env.get_template(f"{template_name}.jinja")
            return template.render(**context)
        except TemplateNotFound:
            raise TemplateNotFound(
                f"Template '{template_name}.jinja' not found in {self.template_dir}"
            )
