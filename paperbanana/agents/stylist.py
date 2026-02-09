"""Stylist Agent: Optimizes diagram descriptions for visual aesthetics."""

from __future__ import annotations

import structlog

from paperbanana.agents.base import BaseAgent
from paperbanana.core.types import DiagramType
from paperbanana.providers.base import VLMProvider

logger = structlog.get_logger()


class StylistAgent(BaseAgent):
    """Refines a textual description to optimize visual aesthetics.

    Takes the Planner's output and enhances it with style-specific
    guidelines while preserving the content.
    """

    def __init__(
        self,
        vlm_provider: VLMProvider,
        guidelines: str = "",
        prompt_dir: str = "prompts",
    ):
        super().__init__(vlm_provider, prompt_dir)
        self.guidelines = guidelines

    @property
    def agent_name(self) -> str:
        return "stylist"

    async def run(
        self,
        description: str,
        guidelines: str | None = None,
        source_context: str = "",
        caption: str = "",
        diagram_type: DiagramType = DiagramType.METHODOLOGY,
    ) -> str:
        """Refine a description for optimal visual aesthetics.

        Args:
            description: The Planner's textual description.
            guidelines: Optional style guidelines (overrides instance default).
            source_context: Original methodology text from the paper.
            caption: Figure caption / communicative intent.
            diagram_type: Type of diagram being generated.

        Returns:
            Stylistically optimized description.
        """
        style_guidelines = guidelines or self.guidelines
        if not style_guidelines:
            style_guidelines = self._default_guidelines()

        prompt_type = "diagram" if diagram_type == DiagramType.METHODOLOGY else "plot"
        template = self.load_prompt(prompt_type)
        prompt = self.format_prompt(
            template,
            description=description,
            guidelines=style_guidelines,
            source_context=source_context,
            caption=caption,
        )

        logger.info("Running stylist agent", description_length=len(description))

        optimized = await self.vlm.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=4096,
        )

        logger.info("Stylist refined description", length=len(optimized))
        return optimized

    def _default_guidelines(self) -> str:
        """Return default aesthetic guidelines if none provided."""
        return """
## Excalidraw-Style Technical Diagram Guidelines

### Drawing Feel
- Use an Excalidraw-like hand-drawn technical style
- Slightly rough strokes, simple shapes, clean white background
- Flat fills only; no gradients, no 3D rendering, no photorealism

### Color Philosophy
- Use soft, muted tones with limited palette (3-5 main hues)
- Keep fills light for readability and use darker matching borders
- Describe colors in natural language only (no hex codes)

### Typography
- Clean sans-serif labels with high legibility
- Keep wording concise and unambiguous
- All text must be clear, readable English

### Layout and Flow
- One clear flow direction (left-to-right or top-to-bottom)
- Consistent spacing and alignment on an implicit grid
- Use grouped containers for phases; avoid clutter

### Connectors and Symbols
- Clear arrows with consistent style and directionality
- Dashed connectors only for optional/auxiliary relations
- Simple iconography only when semantically necessary
"""
