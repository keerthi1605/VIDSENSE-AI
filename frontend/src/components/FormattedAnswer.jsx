// Shows the language model's reply the way it was written: paragraphs, "-" / "*" bullet lists,
// "1." numbered lists, **bold** and `code`. The text is turned into React elements, never into
// HTML, so nothing in a reply can inject markup.

function inline(text, keyBase) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
    if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${keyBase}-${i}`}>{part.slice(2, -2)}</strong>;
    }
    if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) {
      return <code key={`${keyBase}-${i}`}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

const BULLET = /^\s*[-*•]\s+(.*)$/;
const NUMBERED = /^\s*(\d+)[.)]\s+(.*)$/;
const HEADING = /^\s*#{1,6}\s+(.*)$/;

function parse(text) {
  // Small models sometimes run a numbered list together on one line ("1. **A**: x 2. **B**: y");
  // break those apart before reading line by line.
  const spaced = text.replace(/\r/g, "").replace(/([^\n])[ \t]+(?=\d+\.\s+\*\*)/g, "$1\n");
  const blocks = [];
  let paragraph = [];
  let list = null;

  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ type: "p", text: paragraph.join(" ") });
    paragraph = [];
  };
  const flushList = () => {
    if (list) blocks.push(list);
    list = null;
  };

  for (const line of spaced.split("\n")) {
    const bullet = line.match(BULLET);
    const numbered = line.match(NUMBERED);
    const heading = line.match(HEADING);
    if (!line.trim()) {
      flushParagraph();
      flushList();
    } else if (bullet || numbered) {
      flushParagraph();
      const type = numbered ? "ol" : "ul";
      if (!list || list.type !== type) {
        flushList();
        list = { type, items: [] };
      }
      list.items.push(numbered ? numbered[2] : bullet[1]);
    } else if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ type: "h", text: heading[1] });
    } else {
      flushList();
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  flushList();
  return blocks;
}

export default function FormattedAnswer({ text }) {
  return (
    <div className="answer-text">
      {parse(text || "").map((b, i) => {
        if (b.type === "ul" || b.type === "ol") {
          const List = b.type;
          return (
            <List key={i}>
              {b.items.map((item, j) => (
                <li key={j}>{inline(item, `${i}-${j}`)}</li>
              ))}
            </List>
          );
        }
        if (b.type === "h") return <p key={i}><strong>{inline(b.text, String(i))}</strong></p>;
        return <p key={i}>{inline(b.text, String(i))}</p>;
      })}
    </div>
  );
}
