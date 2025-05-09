const handleDownload = async (format) => {
    try {
      if (format === "pdf") {
        const doc = new jsPDF();
        const lines = doc.splitTextToSize(generatedStrategy, 180);
        let y = 20;
 
        lines.forEach((line) => {
          if (y > 280) {
            doc.addPage();
            y = 20;
          }
          doc.text(line, 20, y);
          y += 7;
        });
 
        doc.save("campaign_strategy.pdf");
      } else if (format === "docx") {
        const doc = new Document({
          sections: [
            {
              properties: {},
              children: generatedStrategy.split("\n").map(
                (line) =>
                  new Paragraph({
                    children: [new TextRun(line)],
                  })
              ),
            },
          ],
        });
 
        const blob = await Packer.toBlob(doc);
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "campaign_strategy.docx";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
      } else {
        const element = document.createElement("a");
        const file = new Blob([generatedStrategy], { type: "text/plain" });
        element.href = URL.createObjectURL(file);
        element.download = `campaign_strategy.${format}`;
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
      }
    } catch (error) {
      console.error("Download error:", error);
      alert("Failed to generate document. Please try again.");
    }
  };
 
 