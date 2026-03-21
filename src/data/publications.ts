export interface Publication {
  abbr: string;
  title: string;
  authors: string[];
  journal?: string;
  school?: string;
  booktitle?: string;
  year: string;
  doi?: string;
  link: string;
  selected?: boolean;
  isThesis?: boolean;
}

export const papers: Publication[] = [
  {
    abbr: "bioRxiv",
    title: "A foundational model for joint sequence-function multi-species modeling at scale for long-range genomic prediction",
    authors: [
      "Sam Boshar",
      "Benjamin Evans",
      "Ziqi Tang",
      "Armand Picard",
      "Yanis Adel",
      "Franziska K. Lorbeer",
      "Chandana Rajesh",
      "Tristan Karch",
      "Shawn Sidbon",
      "David Emms",
      "Javier Mendoza-Revilla",
      "Fatimah Al-Ani",
      "Evan Seitz",
      "Yair Schiff",
      "Yohan Bornachot",
      "Ariana Hernandez",
      "Marie Lopez",
      "Alexandre Laterre",
      "Karim Beguir",
      "Peter Koo",
      "Volodymyr Kuleshov",
      "Alexander Stark",
      "Bernardo P. de Almeida",
      "Thomas Pierrot"
    ],
    journal: "bioRxiv",
    year: "2025",
    link: "https://www.biorxiv.org/content/10.64898/2025.12.22.695963v1",
    selected: true
  },
  {
    abbr: "Oxf. Bio",
    title: "Are genomic language models all you need? Exploring genomic language models on protein downstream tasks",
    authors: [
      "Sam Boshar",
      "Evan Trop",
      "Bernardo P. de Almeida",
      "Liviu Copoiu",
      "Thomas Pierrot"
    ],
    journal: "Bioinformatics",
    year: "2024",
    doi: "10.1093/bioinformatics/btae529",
    link: "https://academic.oup.com/bioinformatics/article/40/9/btae529/7745814",
    selected: true
  },
  {
    abbr: "thesis",
    title: "Genomic Language Models for Protein Function and Property Prediction",
    authors: ["Sam Boshar"],
    school: "Massachusetts Institute of Technology",
    year: "2024",
    link: "https://dspace.mit.edu/handle/1721.1/156816",
    isThesis: true
  }
];

export const workshops: Publication[] = [
  {
    abbr: "ICLR",
    title: "Are Genomic Language Models All You Need?",
    authors: [
      "S. Boshar",
      "Trop E.",
      "T. Pierrot",
      "B. Almeida"
    ],
    booktitle: "MLGenX: Machine Learning for Genomics Exploration",
    year: "2024",
    link: "#"
  }
];

const NAMES_TO_BOLD = new Set(["Sam Boshar", "S. Boshar"]);

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function authorToHtml(name: string): string {
  const safe = escapeHtml(name);
  if (NAMES_TO_BOLD.has(name)) {
    return `<strong class="font-semibold">${safe}</strong>`;
  }
  return safe;
}

/** Comma-separated author list HTML; your name is wrapped in &lt;strong&gt;. */
export function formatAuthors(
  authors: string[],
  maxVisible: number = 3
): { html: string; hasMore: boolean; moreCount: number } {
  if (authors.length <= maxVisible) {
    return {
      html: authors.map(authorToHtml).join(", "),
      hasMore: false,
      moreCount: 0,
    };
  }
  const visible = authors.slice(0, maxVisible);
  const remaining = authors.length - maxVisible;
  return {
    html: `${visible.map(authorToHtml).join(", ")}, and `,
    hasMore: true,
    moreCount: remaining,
  };
}
