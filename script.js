// Function to show the info page with README.md content
function showInfoPage() {
  const infoOverlay = document.getElementById('info-overlay');
  const infoContent = document.getElementById('info-content');
  
  // Show the overlay
  infoOverlay.style.display = 'flex';
  
  // Load README.md content if not already loaded
  if (!infoContent.innerHTML.trim()) {
    fetch('README.md')
      .then(response => response.text())
      .then(markdown => {
        // Simple markdown to HTML conversion
        let html = markdown
          // Convert headers
          .replace(/^# (.*$)/gm, '<h1>$1</h1>')
          .replace(/^## (.*$)/gm, '<h2>$1</h2>')
          .replace(/^### (.*$)/gm, '<h3>$1</h3>')
          
          // Convert links
          .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
          
          // Convert lists
          .replace(/^\s*-\s+(.*$)/gm, '<li>$1</li>')
          .replace(/(<li>.*<\/li>\n)+/g, '<ul>$&</ul>')
          
          // Convert emphasis
          .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
          .replace(/\*([^*]+)\*/g, '<em>$1</em>')
          
          // Convert code blocks
          .replace(/`([^`]+)`/g, '<code>$1</code>')
          
          // Convert paragraphs (lines that are not headers, lists, or blank)
          .replace(/^(?!<h|<ul|<li|$)(.*$)/gm, '<p>$1</p>')
          
          // Handle color spans for distinction focus (gold) and capability descriptions (white on blue)
          .replace(/<span style="color:#D9A441;[^>]*>([^<]+)<\/span>/g, 
                   '<span style="color:#D9A441; font-weight: bold;">$1</span>')
          .replace(/<span style="color:#FFFFFF; background-color:#4575B4;[^>]*>([^<]+)<\/span>/g, 
                   '<span style="color:#FFFFFF; background-color:#4575B4; padding: 2px 4px; border-radius: 3px;">$1</span>');
        
        infoContent.innerHTML = html;
      })
      .catch(error => {
        infoContent.innerHTML = '<p>Error loading README.md: ' + error.message + '</p>';
      });
  }
}

// Expanded list of English stop words, including technical and academic contexts
const STOP_WORDS = new Set([
  // Basic articles and pronouns
  'a', 'an', 'the', 'this', 'that', 'these', 'those', 
  'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
  'you', 'your', 'yours', 'yourself', 'yourselves',
  'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
  'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',

  // Prepositions and conjunctions
  'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'as', 'if', 'because',
  'since', 'unless', 'until', 'while', 'although', 'though', 'despite',
  'in', 'on', 'at', 'to', 'from', 'with', 'by', 'about', 'under', 'over',
  'between', 'among', 'through', 'across', 'into', 'onto', 'toward',

  // Auxiliary and modal verbs
  'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being',
  'have', 'has', 'had', 'having',
  'do', 'does', 'did', 'doing',
  'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',

  // Additional common words
  'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
  'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
  'just', 'now', 'how', 'when', 'where', 'why', 'what', 'which', 'who', 'whom'
]);

// Expanded technical term mappings with more comprehensive coverage
const TECH_TERMS = {
  'py': 'python',
  'js': 'javascript',
  'cpp': 'c++',
  'cs': 'c sharp',
  'rb': 'ruby',
  'php': 'php',
  'java': 'java',
  'kt': 'kotlin',
  'swift': 'swift',
  'rs': 'rust',
  'go': 'golang',

  'ml': 'machine learning',
  'ai': 'artificial intelligence',
  'dl': 'deep learning',
  'nn': 'neural network',
  'nlp': 'natural language processing',
  'cv': 'computer vision',
  'rl': 'reinforcement learning',
  'gan': 'generative adversarial network',

  'db': 'database',
  'sql': 'structured query language',
  'nosql': 'not only sql',
  'olap': 'online analytical processing',
  'etl': 'extract transform load',
  'dw': 'data warehouse',
  'ds': 'data structure',
  'algo': 'algorithm',

  'api': 'application programming interface',
  'ui': 'user interface',
  'ux': 'user experience',
  'cms': 'content management system',
  'crm': 'customer relationship management',
  'saas': 'software as a service',
  'paas': 'platform as a service',
  'iaas': 'infrastructure as a service',
  'oop': 'object oriented programming',
  'fp': 'functional programming',

  'aws': 'amazon web services',
  'gcp': 'google cloud platform',
  'azure': 'microsoft azure',
  'k8s': 'kubernetes',
  'ci': 'continuous integration',
  'cd': 'continuous deployment',
  'vpc': 'virtual private cloud',

  'vpn': 'virtual private network',
  'ssl': 'secure sockets layer',
  'dns': 'domain name system',
  'http': 'hypertext transfer protocol',
  'https': 'hypertext transfer protocol secure',
  'ip': 'internet protocol',

  'auth': 'authentication',
  'regex': 'regular expression',
  'img': 'image',
  'config': 'configuration'
};


// Porter Stemmer implementation in JavaScript
class PorterStemmer {
  constructor() {
    this.cache = new Map();
  }

  stem(word) {
    if (this.cache.has(word)) {
      return this.cache.get(word);
    }

    let stem = word;

    // Step 1a
    const step1aRegex = /^(.+?)(ss|i)es$/;
    const step1aRegex2 = /^(.+?)([^s])s$/;
    if (step1aRegex.test(word)) {
      stem = word.replace(step1aRegex, '$1$2');
    } else if (step1aRegex2.test(word)) {
      stem = word.replace(step1aRegex2, '$1$2');
    }

    // Step 1b
    if (/^(.+?)eed$/.test(stem)) {
      const fp = /^(.+?)eed$/.exec(stem);
      if (/[aeiou].*[^aeiou][^aeiou].*/.test(fp[1])) {
        stem = stem.slice(0, -1);
      }
    } else {
      let regex = /^(.+?)ed$/;
      let fp = regex.exec(stem);
      if (fp) {
        if (/[aeiou].*[^aeiou]/.test(fp[1])) {
          stem = fp[1];
          if (/at$|bl$|iz$/.test(stem)) {
            stem += 'e';
          } else if (/([^aeiouylsz])\1$/.test(stem)) {
            stem = stem.slice(0, -1);
          } else if (/^[^aeiou][^aeiou][aeiou].*[^aeiouwxy]$/.test(stem)) {
            stem += 'e';
          }
        }
      } else {
        regex = /^(.+?)ing$/;
        fp = regex.exec(stem);
        if (fp && /[aeiou].*[^aeiou]/.test(fp[1])) {
          stem = fp[1];
          if (/at$|bl$|iz$/.test(stem)) {
            stem += 'e';
          } else if (/([^aeiouylsz])\1$/.test(stem)) {
            stem = stem.slice(0, -1);
          } else if (/^[^aeiou][^aeiou][aeiou].*[^aeiouwxy]$/.test(stem)) {
            stem += 'e';
          }
        }
      }
    }

    // Step 1c
    if (/^(.+?)y$/.test(stem)) {
      const fp = /^(.+?)y$/.exec(stem);
      if (/[aeiou].*[^aeiou]/.test(fp[1])) {
        stem = fp[1] + 'i';
      }
    }

    this.cache.set(word, stem);
    return stem;
  }
}

class BM25Search {
  constructor(k1 = 1.2, b = 0.75) {
    this.k1 = k1;
    this.b = b;
    this.reset();
    this.stemmer = new PorterStemmer();
  }

  reset() {
    this.documents = [];
    this.docFreq = new Map();
    this.docLengths = [];
    this.avgDocLength = 0;
    this.totalDocs = 0;
    this.nodeInfo = [];
    this.phraseIndex = new Map();
  }

  expandTechnicalTerms(text) {
    const words = text.toLowerCase().split(/\s+/);
    const expanded = [];
    let i = 0;
    while (i < words.length) {
      if (i < words.length - 1) {
        const twoWord = `${words[i]} ${words[i + 1]}`;
        if (TECH_TERMS[twoWord]) {
          expanded.push(TECH_TERMS[twoWord]);
          i += 2;
          continue;
        }
      }
      if (TECH_TERMS[words[i]]) {
        expanded.push(TECH_TERMS[words[i]]);
      } else {
        expanded.push(words[i]);
      }
      i++;
    }
    return expanded.join(' ');
  }

  extractPhrases(text) {
    const phrases = text.match(/"([^"]*)"/g)?.map(p => p.slice(1, -1)) || [];
    const remaining = text.replace(/"[^"]*"/g, '');
    return [phrases, remaining];
  }

  tokenize(text, forIndexing = false) {
    if (!text || typeof text !== 'string') return [];
    
    // Limit text length to prevent RangeError
    text = text.slice(0, 10000);
    text = this.expandTechnicalTerms(text);

    let tokens = [];
    if (!forIndexing) {
      const [phrases, words] = this.extractPhrases(text);
      phrases.forEach(phrase => {
        if (phrase && phrase.length < 1000) {  // Add length check for phrases
          const normalized = phrase.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').trim();
          if (normalized) tokens.push(normalized);
        }
      });
      text = words;
    }

    text = text.toLowerCase().replace(/[^a-z0-9\s-]/g, ' ');
    
    const words = [];
    const textWords = text.split(/\s+/);
    for (let i = 0; i < Math.min(textWords.length, 1000); i++) {  // Limit number of words
      const word = textWords[i];
      if (!word || word.length > 100) continue;  // Skip very long words
      
      if (word.includes('-')) {
        const parts = word.split('-');
        if (parts.length < 10) {  // Prevent excessive hyphenation
          words.push(...parts);
          if (parts.join('').length > 3) {
            words.push(word);
          }
        }
      } else {
        words.push(word);
      }
    }

    tokens = [...tokens, ...words.filter(word => 
      word && !STOP_WORDS.has(word) && word.length > 1 && word.length < 100  // Add length check
    ).map(word => this.stemmer.stem(word))];

    if (forIndexing && tokens.length) {
      // Limit number of n-grams to prevent memory issues
      const maxNGrams = 1000;
      const nGrams = [];
      
      for (let i = 0; i < Math.min(tokens.length - 1, maxNGrams); i++) {
        nGrams.push(`${tokens[i]} ${tokens[i + 1]}`);
        if (i < tokens.length - 2) {
          nGrams.push(`${tokens[i]} ${tokens[i + 1]} ${tokens[i + 2]}`);
        }
      }
      
      tokens = [...tokens, ...nGrams];
    }

    // Final safety check
    return tokens.slice(0, 5000);
  }

  traverseTree(node, currentPath = []) {
    if ("capability" in node) {
      const idx = this.documents.length;
      const isLeaf = !("subtrees" in node && Array.isArray(node.subtrees));
      const input = isLeaf ? node.input || "" : "";
      const text = isLeaf ? `${node.capability} ${input}` : node.capability;
      const tokens = this.tokenize(text, true);

      this.documents.push(tokens);
      this.docLengths.push(tokens.length);
      this.nodeInfo.push({
        capability: node.capability,
        path: [...currentPath],
        depth: currentPath.length,
        input: input,
        isLeaf: isLeaf
      });

      // Index phrases
      tokens.forEach(token => {
        if (!this.phraseIndex.has(token)) {
          this.phraseIndex.set(token, []);
        }
        this.phraseIndex.get(token).push(idx);
      });

      // Update document frequencies
      new Set(tokens).forEach(token => {
        this.docFreq.set(token, (this.docFreq.get(token) || 0) + 1);
      });
    }

    if ("subtrees" in node && Array.isArray(node.subtrees)) {
      node.subtrees.forEach((subtree, i) => {
        currentPath.push(i);
        this.traverseTree(subtree, currentPath);
        currentPath.pop();
      });
    }
  }

  buildIndex(treeData) {
    this.reset();
    this.traverseTree(treeData);
    this.totalDocs = this.documents.length;
    this.avgDocLength = this.docLengths.reduce((a, b) => a + b, 0) / this.totalDocs;
  }

  calculateBM25Score(query, docIndex) {
    const doc = this.documents[docIndex];
    const docLength = this.docLengths[docIndex];
    let score = 0;

    const queryTerms = new Set(query);
    queryTerms.forEach(term => {
      const tf = doc.filter(t => t === term).length;
      const df = this.docFreq.get(term) || 0;
      if (df > 0) {
        const idf = Math.log((this.totalDocs - df + 0.5) / (df + 0.5) + 1);
        score += idf * ((tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + this.b * docLength / this.avgDocLength)));
      }
    });

    return score;
  }

  calculateFinalScore(bm25Score, depth, hasPhraseMatch) {
    const depthFactor = 1.0 / (1.0 + 0.1 * depth);
    const phraseBoost = hasPhraseMatch ? 1.5 : 1.0;
    return bm25Score * depthFactor * phraseBoost;
  }

  search(query, topK = 10) {
    const [phrases, remainingWords] = this.extractPhrases(query);
    const queryTokens = this.tokenize(remainingWords);

    if (!queryTokens.length && !phrases.length) {
      return [];
    }

    // Calculate BM25 scores
    let bm25Scores = new Array(this.totalDocs).fill(0);
    if (queryTokens.length) {
      bm25Scores = bm25Scores.map((_, i) => this.calculateBM25Score(queryTokens, i));
    }

    // Normalize BM25 scores
    const maxBM25 = Math.max(...bm25Scores) || 1;
    bm25Scores = bm25Scores.map(score => score / maxBM25);

    // Find phrase matches
    const phraseMatches = new Set();
    phrases.forEach(phrase => {
      const normalizedPhrase = this.tokenize(phrase, false).join(' ');
      if (this.phraseIndex.has(normalizedPhrase)) {
        this.phraseIndex.get(normalizedPhrase).forEach(idx => phraseMatches.add(idx));
      }
    });

    // Calculate final scores
    const minScoreThreshold = 0.3;
    const results = [];

    bm25Scores.forEach((bm25Score, idx) => {
      const hasPhraseMatch = phraseMatches.has(idx);
      const finalScore = this.calculateFinalScore(
        bm25Score,
        this.nodeInfo[idx].depth,
        hasPhraseMatch
      );

      if (finalScore > minScoreThreshold) {
        results.push({
          ...this.nodeInfo[idx],
          score: finalScore
        });
      }
    });

    // Sort by depth first, then by score
    results.sort((a, b) => a.depth !== b.depth ? a.depth - b.depth : b.score - a.score);

    return results.slice(0, topK);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  // Set up info overlay event listeners
  const infoOverlay = document.getElementById('info-overlay');
  
  // Add close button functionality
  const infoCloseButton = infoOverlay.querySelector('.details-close');
  infoCloseButton.onclick = () => {
    infoOverlay.style.display = 'none';
  };
  
  // Close when clicking outside the content
  infoOverlay.onclick = (event) => {
    if (event.target === infoOverlay) {
      infoOverlay.style.display = 'none';
    }
  };
  
  // Automatically open info page on first visit
  if (!sessionStorage.getItem('infoPageShown')) {
    showInfoPage();
    sessionStorage.setItem('infoPageShown', 'true');
  }
  
  // Global ESC key handler for all overlays and UI elements
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      // Close info overlay if it's open
      if (infoOverlay.style.display === 'flex') {
        infoOverlay.style.display = 'none';
        return; // Only handle one action per ESC press
      }
      
      // Close details overlay if it's open
      const detailsOverlay = document.getElementById('details-overlay');
      if (detailsOverlay.style.display === 'flex') {
        detailsOverlay.style.display = 'none';
        return; // Only handle one action per ESC press
      }
      
      // Clear search input and hide results if focused
      // (The search-specific handler will still work when the search input is focused)
      const searchInput = document.getElementById('search-input');
      const searchResults = document.getElementById('search-results');
      if (searchResults && searchResults.style.display === 'block' && !document.activeElement.isEqualNode(searchInput)) {
        searchInput.value = '';
        searchResults.style.display = 'none';
      }
    }
  });
  // Track loading state
  let isLoading = false;
  let currentLoadingAbortController = null;
  let searchEngine = new BM25Search();

  function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
  // Add search UI elements
  const searchContainer = document.createElement('div');
  searchContainer.style.cssText = `
    position: fixed;
    top: 200px;
    left: 20px;
    z-index: 2001;
    background: white;
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    width: 400px;
  `;

  const searchInput = document.createElement('input');
  searchInput.type = 'text';
  searchInput.placeholder = 'Type to search tree node capabilities and press ENTER';
  searchInput.style.cssText = `
    width: 92%;
    padding: 12px;
    border: 2px solid #2196f3;
    border-radius: 6px;
    margin-bottom: 10px;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s;
  `;
  
  // Add focus effect
  searchInput.addEventListener('focus', () => {
    searchInput.style.borderColor = '#1976d2';
  });
  
  searchInput.addEventListener('blur', () => {
    searchInput.style.borderColor = '#2196f3';
  });

  const searchResults = document.createElement('div');
  searchResults.style.cssText = `
    max-height: 600px;
    overflow-y: auto;
    display: none;
  `;

  searchContainer.appendChild(searchInput);
  searchContainer.appendChild(searchResults);
  document.body.appendChild(searchContainer);
  
  // Prevent clicks inside search container from closing the search results
  searchContainer.addEventListener('click', (e) => {
    e.stopPropagation();
  });
  
  // Close search results when clicking outside the search container
  document.addEventListener('click', () => {
    if (searchResults.style.display === 'block') {
      searchResults.style.display = 'none';
    }
  });

  // Load meta.json to get dataset information
  fetch('meta.json')
    .then(response => response.json())
    .then(datasets => {
      // Create dataset buttons container
      // Create a header container for title and buttons
      const headerContainer = document.createElement('div');
      headerContainer.style.cssText = `
        display: flex;
        flex-direction: column;
        padding: 10px 20px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        overflow: hidden;
        min-height: 100px;
      `;
      
      document.body.prepend(headerContainer);
      
      // Create top row for title and dataset buttons
      const topRow = document.createElement('div');
      topRow.style.cssText = `
        display: flex;
        align-items: center;
        margin-bottom: 10px;
      `;
      
      // Add a wrapper for the info button and title to prevent it from shrinking
      const titleWrapper = document.createElement('div');
      titleWrapper.style.cssText = `
        flex: 0 0 auto;
        display: flex;
        align-items: center;
      `;
      
      // Create info button
      const infoButton = document.createElement('button');
      infoButton.className = 'info-button';
      infoButton.innerHTML = '<i>ℹ</i><span>How it Works</span>';
      infoButton.title = 'How it Works';
      infoButton.addEventListener('click', showInfoPage);
      titleWrapper.appendChild(infoButton);
      
      // Create a title element in the header
      const titleElement = document.createElement('h1');
      titleElement.style.margin = '0 20px 0 0';
      titleElement.style.padding = '0';
      titleElement.style.boxShadow = 'none';
      
      // Create a clickable link for the dataset title
      const titleLink = document.createElement('a');
      titleLink.id = 'header-title';
      titleLink.href = datasets['MATH'].url; // Default URL
      
      // Create a container for the title text and icon
      const titleContainer = document.createElement('span');
      titleContainer.style.display = 'inline-flex';
      titleContainer.style.alignItems = 'center';
      
      // Add the title text
      const titleText = document.createElement('span');
      titleText.textContent = 'MATH'; // Default title
      titleContainer.appendChild(titleText);
      
      // Add an external link icon
      const externalLinkIcon = document.createElement('span');
      externalLinkIcon.innerHTML = ' &#x2197;'; // Unicode for north-east arrow
      externalLinkIcon.style.fontSize = '0.7em';
      externalLinkIcon.style.marginLeft = '5px';
      externalLinkIcon.style.verticalAlign = 'super';
      titleContainer.appendChild(externalLinkIcon);
      
      titleLink.appendChild(titleContainer);
      
      // Style the link to make it clear it's clickable
      titleLink.style.color = '#2196f3'; // Blue color to indicate a link
      titleLink.style.textDecoration = 'none'; // Start without underline
      titleLink.style.borderBottom = '1px solid #2196f3'; // Add bottom border instead
      titleLink.style.transition = 'all 0.2s ease';
      titleLink.style.cursor = 'pointer'; // Add pointer cursor
      
      // Hover effects
      titleLink.onmouseover = () => { 
        titleLink.style.textDecoration = 'none';
        titleLink.style.borderBottom = '2px solid #2196f3';
        titleLink.style.opacity = '0.8';
      };
      titleLink.onmouseout = () => { 
        titleLink.style.textDecoration = 'none';
        titleLink.style.borderBottom = '1px solid #2196f3';
        titleLink.style.opacity = '1';
      };
      
      titleElement.appendChild(titleLink);
      titleWrapper.appendChild(titleElement);
      topRow.appendChild(titleWrapper);
      
      // Create guidance box
      const guidanceBox = document.createElement('div');
      guidanceBox.style.cssText = `
        background-color: #f0f6fa;
        border: 1px solid #92C5E0;
        border-radius: 8px;
        padding: 10px 15px;
        margin-right: 15px;
        font-size: 13px;
        line-height: 1.5;
        color: var(--text-dark);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        max-width: 320px;
        flex-shrink: 0;
      `;
      
      // Add two-line guidance text
      const guidanceLine1 = document.createElement('div');
      guidanceLine1.innerHTML = "<strong>Left-click</strong> a non-leaf node to expand/collapse it.";
      guidanceLine1.style.marginBottom = "4px";
      
      const guidanceLine2 = document.createElement('div');
      guidanceLine2.innerHTML = "<strong>Right-click</strong> a node to view detailed information.";
      
      guidanceBox.appendChild(guidanceLine1);
      guidanceBox.appendChild(guidanceLine2);
      
      topRow.appendChild(guidanceBox);
      
      // Create a wrapper for benchmark section with title
      const benchmarkSectionWrapper = document.createElement('div');
      benchmarkSectionWrapper.style.cssText = `
        flex: 1;
        display: flex;
        align-items: center;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 5px 10px;
        margin-right: 10px;
        background-color: #f9f9f9;
        overflow: hidden; /* Prevent wrapper from expanding */
      `;
      
      // Create benchmark title
      const benchmarkTitle = document.createElement('div');
      benchmarkTitle.textContent = 'Benchmark';
      benchmarkTitle.style.cssText = `
        font-weight: 600;
        margin-right: 15px;
        white-space: nowrap;
        color: #555;
        font-size: 14px;
      `;
      
      // Create a scrollable container for dataset buttons
      const datasetButtonScrollContainer = document.createElement('div');
      datasetButtonScrollContainer.style.cssText = `
        flex: 1;
        min-width: 0; /* Allow container to shrink */
        overflow-x: auto;
        white-space: nowrap;
        padding: 5px 0;
        scrollbar-width: thin;
        scrollbar-color: #ccc transparent;
      `;
      
      // Add custom scrollbar styling
      datasetButtonScrollContainer.innerHTML = `
        <style>
          .button-scroll-container::-webkit-scrollbar {
            height: 6px;
          }
          .button-scroll-container::-webkit-scrollbar-track {
            background: transparent;
          }
          .button-scroll-container::-webkit-scrollbar-thumb {
            background-color: #ccc;
            border-radius: 6px;
          }
        </style>
      `;
      
      datasetButtonScrollContainer.className = 'button-scroll-container';
      
      // Create button container for dataset selection
      const datasetButtonContainer = document.createElement('div');
      datasetButtonContainer.style.cssText = `
        display: inline-flex;
        gap: 10px;
        align-items: center;
        padding: 5px 0;
        flex-wrap: nowrap; /* Prevent wrapping */
      `;
      
      datasetButtonScrollContainer.appendChild(datasetButtonContainer);
      benchmarkSectionWrapper.appendChild(benchmarkTitle);
      benchmarkSectionWrapper.appendChild(datasetButtonScrollContainer);
      topRow.appendChild(benchmarkSectionWrapper);
      
      // Add the top row to the header container
      headerContainer.appendChild(topRow);
      
      // Create bottom row for model buttons
      const bottomRow = document.createElement('div');
      bottomRow.style.cssText = `
        display: flex;
        align-items: center;
      `;
      
      // Create a wrapper for model section with title
      const modelSectionWrapper = document.createElement('div');
      modelSectionWrapper.style.cssText = `
        flex: 1;
        display: flex;
        align-items: center;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 5px 10px;
        background-color: #f9f9f9;
        overflow: hidden; /* Prevent wrapper from expanding */
      `;
      
      // Create model title
      const modelTitle = document.createElement('div');
      modelTitle.textContent = 'Model';
      modelTitle.style.cssText = `
        font-weight: 600;
        margin-right: 15px;
        white-space: nowrap;
        color: #555;
        font-size: 14px;
      `;
      
      // Create a scrollable container for model buttons
      const modelButtonScrollContainer = document.createElement('div');
      modelButtonScrollContainer.style.cssText = `
        flex: 1;
        min-width: 0; /* Allow container to shrink */
        overflow-x: auto;
        white-space: nowrap;
        padding: 5px 0;
        scrollbar-width: thin;
        scrollbar-color: #ccc transparent;
      `;
      
      modelButtonScrollContainer.className = 'button-scroll-container';
      
      // Create button container for model selection
      const modelButtonContainer = document.createElement('div');
      modelButtonContainer.id = 'model-button-container';
      modelButtonContainer.style.cssText = `
        display: inline-flex;
        gap: 10px;
        align-items: center;
        padding: 5px 0;
        flex-wrap: nowrap; /* Prevent wrapping */
      `;
      
      modelButtonScrollContainer.appendChild(modelButtonContainer);
      modelSectionWrapper.appendChild(modelTitle);
      modelSectionWrapper.appendChild(modelButtonScrollContainer);
      bottomRow.appendChild(modelSectionWrapper);
      
      // Add the bottom row to the header container
      headerContainer.appendChild(bottomRow);
      
      // We're using the global currentModel variable
      
      // Function to create model buttons for a dataset
      function createModelButtons(datasetName) {
        // Clear existing model buttons
        modelButtonContainer.innerHTML = '';
        
        // Create "(NONE)" button first with distinct styling
        const allModelsButton = document.createElement('button');
        allModelsButton.textContent = '(NONE)';
        allModelsButton.className = 'button pressed';
        allModelsButton.style.minWidth = '140px';  // Wider than regular buttons
        allModelsButton.style.margin = '4px 8px 4px 4px';  // Extra right margin for separation
        allModelsButton.style.whiteSpace = 'nowrap';
        allModelsButton.style.flex = '0 0 auto';
        allModelsButton.style.backgroundColor = '#3f51b5';
        allModelsButton.style.color = 'white';
        allModelsButton.style.fontWeight = '600';  // Make text bolder
        allModelsButton.style.fontSize = '15px';   // Slightly larger text
        allModelsButton.style.padding = '14px 20px';  // More padding for larger appearance
        allModelsButton.style.borderRadius = '8px';  // Rounded corners
        allModelsButton.style.border = '2px solid #3f51b5';  // Matching border
        allModelsButton.style.boxShadow = '0 3px 6px rgba(0,0,0,0.16)';  // Subtle shadow for depth
        allModelsButton.onclick = () => {
          if (isLoading) {
            return; // Prevent switching while loading
          }
          
          // Remove pressed state from all model buttons
          document.querySelectorAll('#model-button-container .button').forEach(btn => {
            btn.classList.remove('pressed');
            
            // Special handling for (NONE) button to maintain its distinct appearance
            if (btn === allModelsButton) {
              btn.style.backgroundColor = 'white';
              btn.style.color = 'var(--primary-color)';
              btn.style.fontWeight = '600';
              btn.style.border = '2px solid #e0e0e0';
              btn.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
            } else {
              // Regular model buttons
              btn.style.backgroundColor = 'white';
              btn.style.color = 'var(--primary-color)';
              btn.style.border = '1px solid #e0e0e0';
            }
          });
          
          // Add pressed state to this button
          allModelsButton.classList.add('pressed');
          allModelsButton.style.backgroundColor = '#3f51b5';
          allModelsButton.style.color = 'white';
          allModelsButton.style.fontWeight = '600';
          allModelsButton.style.border = '2px solid #3f51b5';
          allModelsButton.style.boxShadow = '0 3px 6px rgba(0,0,0,0.16)';
          
          // Reset current model
          currentModel = null;
          
          // Refresh the tree visualization
          loadTreeData(datasets[datasetName].path);
        };
        
        // Add to container
        modelButtonContainer.appendChild(allModelsButton);
        
        // Create buttons for each model in the dataset
        if (datasets[datasetName].models && datasets[datasetName].models.length > 0) {
          datasets[datasetName].models.forEach(modelName => {
            const button = document.createElement('button');
            button.textContent = modelName;
            button.className = 'button';
            button.style.minWidth = '100px';
            button.style.margin = '4px';
            button.style.whiteSpace = 'nowrap';
            button.style.flex = '0 0 auto';
            button.style.padding = '10px 16px';  // Smaller padding than (NONE)
            button.style.borderRadius = '6px';   // Slightly less rounded than (NONE)
            button.style.border = '1px solid #e0e0e0';  // Lighter border
            button.style.fontSize = '14px';      // Smaller font than (NONE)
            button.onclick = () => {
              if (isLoading) {
                return; // Prevent switching while loading
              }
              
              // Remove pressed state from all model buttons
              document.querySelectorAll('#model-button-container .button').forEach(btn => {
                btn.classList.remove('pressed');
                
                // Special handling for (NONE) button to maintain its distinct appearance
                if (btn === allModelsButton) {
                  btn.style.backgroundColor = 'white';
                  btn.style.color = 'var(--primary-color)';
                  btn.style.fontWeight = '600';
                  btn.style.border = '2px solid #e0e0e0';
                  btn.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                } else {
                  // Regular model buttons
                  btn.style.backgroundColor = 'white';
                  btn.style.color = 'var(--primary-color)';
                  btn.style.border = '1px solid #e0e0e0';
                }
              });
              
              // Add pressed state to this button
              button.classList.add('pressed');
              button.style.backgroundColor = '#3f51b5';
              button.style.color = 'white';
              button.style.border = '1px solid #3f51b5';  // Matching border color
              
              // Set current model
              currentModel = modelName;
              
              // Refresh the tree visualization
              loadTreeData(datasets[datasetName].path);
            };
            
            // Add to container
            modelButtonContainer.appendChild(button);
          });
        }
      }

      // Create buttons for each dataset
      Object.keys(datasets).forEach(datasetName => {
        const button = document.createElement('button');
        button.textContent = datasetName;
        button.className = 'button';
        button.style.minWidth = '100px';
        button.style.margin = '4px';
        button.style.whiteSpace = 'nowrap';
        button.style.flex = '0 0 auto';
        button.onclick = () => {
          if (isLoading) {
            return; // Prevent switching while loading
          }

          // Update title and its URL
          const titleLink = document.getElementById('header-title');
          titleLink.href = datasets[datasetName].url;
          
          // Clear the title link content and recreate it
          titleLink.innerHTML = '';
          
          // Create a container for the title text and icon
          const titleContainer = document.createElement('span');
          titleContainer.style.display = 'inline-flex';
          titleContainer.style.alignItems = 'center';
          
          // Add the title text
          const titleText = document.createElement('span');
          titleText.textContent = datasetName;
          titleContainer.appendChild(titleText);
          
          // Add an external link icon
          const externalLinkIcon = document.createElement('span');
          externalLinkIcon.innerHTML = ' &#x2197;'; // Unicode for north-east arrow
          externalLinkIcon.style.fontSize = '0.7em';
          externalLinkIcon.style.marginLeft = '5px';
          externalLinkIcon.style.verticalAlign = 'super';
          titleContainer.appendChild(externalLinkIcon);
          
          titleLink.appendChild(titleContainer);
          
          // Remove pressed state from all dataset buttons
          document.querySelectorAll('.button').forEach(btn => {
            btn.classList.remove('pressed');
          });
          
          // Disable all buttons during loading
          document.querySelectorAll('.button').forEach(btn => {
            btn.disabled = true;
          });
          
          // Add pressed state to clicked button
          button.classList.add('pressed');
          
          // Update metrics and load the dataset
          currentMetrics = datasets[datasetName].metrics || "Score";
          
          // Reset current model
          currentModel = null;
          
          // Create model buttons for this dataset
          createModelButtons(datasetName);
          
          // Load the dataset
          loadTreeData(datasets[datasetName].path);
        };
        
        // Add to container
        datasetButtonContainer.appendChild(button);
      });
      
      // Add scroll indicators if needed
      const addScrollIndicators = () => {
        // Remove any existing scroll indicators
        const existingIndicators = headerContainer.querySelectorAll('.scroll-indicator');
        existingIndicators.forEach(indicator => indicator.remove());
        
        // Check if benchmark scrolling is needed
        if (datasetButtonContainer.scrollWidth > datasetButtonScrollContainer.clientWidth) {
          // Add benchmark scroll indicator
          const benchmarkScrollIndicator = document.createElement('div');
          benchmarkScrollIndicator.className = 'scroll-indicator';
          benchmarkScrollIndicator.style.cssText = `
            position: absolute;
            right: 20px;
            top: 40px;
            transform: translateY(-50%);
            font-size: 12px;
            color: #666;
            pointer-events: none;
            z-index: 1001;
          `;
          benchmarkScrollIndicator.textContent = '← scroll →';
          headerContainer.appendChild(benchmarkScrollIndicator);
        }
        
        // Check if model scrolling is needed
        if (modelButtonContainer.scrollWidth > modelButtonScrollContainer.clientWidth) {
          // Add model scroll indicator
          const modelScrollIndicator = document.createElement('div');
          modelScrollIndicator.className = 'scroll-indicator';
          modelScrollIndicator.style.cssText = `
            position: absolute;
            right: 20px;
            top: 85px;
            transform: translateY(-50%);
            font-size: 12px;
            color: #666;
            pointer-events: none;
            z-index: 1001;
          `;
          modelScrollIndicator.textContent = '← scroll →';
          headerContainer.appendChild(modelScrollIndicator);
        }
      };
      
      // Add scroll indicators after buttons are added
      setTimeout(addScrollIndicators, 500);
      
      // Add horizontal scroll event listeners for mouse wheel
      datasetButtonScrollContainer.addEventListener('wheel', (e) => {
        if (e.deltaY !== 0) {
          e.preventDefault();
          datasetButtonScrollContainer.scrollLeft += e.deltaY;
        }
      });
      
      modelButtonScrollContainer.addEventListener('wheel', (e) => {
        if (e.deltaY !== 0) {
          e.preventDefault();
          modelButtonScrollContainer.scrollLeft += e.deltaY;
        }
      });

      // Disable all buttons initially
      document.querySelectorAll('.button').forEach(btn => {
        btn.disabled = true;
      });

      // Load default dataset (MATH)
      const defaultButton = datasetButtonContainer.querySelector('button');
      defaultButton.classList.add('pressed');
      currentMetrics = datasets['MATH'].metrics || "Score";
      
      // Create model buttons for the default dataset
      createModelButtons('MATH');
      
      // Load the dataset
      loadTreeData(datasets['MATH'].path);
    })
    .catch(error => console.error('Error loading datasets:', error));

  let currentMetrics = "Score"; // Default metrics name
  let currentModel = null; // Variable to track the currently selected model

  function loadTreeData(jsonFile) {
    // If there's an ongoing load, abort it
    if (currentLoadingAbortController) {
      currentLoadingAbortController.abort();
    }

    // Clear existing visualization
    d3.select('svg g').remove();

    isLoading = true;
    currentLoadingAbortController = new AbortController();
    
    // Store the fact that we're switching to a new benchmark/model
    // This will be used to reset the view position after loading
    window.resetViewPositionAfterLoad = true;

    let allNodes = []; // Store all nodes for search
    const width = document.querySelector('svg').clientWidth;
    const height = 1500;

    d3.select('svg g').remove();

    // Get the SVG element and its dimensions
    const svgElement = document.querySelector('svg');
    const svgWidth = svgElement.clientWidth || 2500;
    const svgHeight = svgElement.clientHeight || 2000;
    
    // Define constants for the initial position of the tree
    // These values determine where the root node will be positioned
    const INITIAL_X = 768; // Horizontal position (from left)
    const INITIAL_Y = 192; // Vertical position (from top)
    
    // Initialize the SVG with zoom behavior
    const svg = d3.select('svg')
      .attr('viewBox', `0 0 ${svgWidth} ${svgHeight}`)
      .call(d3.zoom().scaleExtent([0.2, 5]).on('zoom', (event) => {
        g.attr('transform', event.transform);
      }));

    // Position the tree with the root node fully below the search box
    // Using the constants defined above for consistent positioning
    const g = svg.append('g').attr('transform', `translate(${INITIAL_X}, ${INITIAL_Y})`);

    // Create filters for different effects
    const defs = svg.append('defs');
    
    // Drop shadow filter
    const dropShadow = defs.append('filter')
      .attr('id', 'drop-shadow')
      .attr('height', '130%');

    dropShadow.append('feGaussianBlur')
      .attr('in', 'SourceAlpha')
      .attr('stdDeviation', 4)
      .attr('result', 'blur');

    dropShadow.append('feOffset')
      .attr('in', 'blur')
      .attr('dx', 2)
      .attr('dy', 3)
      .attr('result', 'offsetBlur');

    dropShadow.append('feComponentTransfer')
      .append('feFuncA')
      .attr('type', 'linear')
      .attr('slope', 0.5);

    const dropShadowMerge = dropShadow.append('feMerge');
    dropShadowMerge.append('feMergeNode').attr('in', 'offsetBlur');
    dropShadowMerge.append('feMergeNode').attr('in', 'SourceGraphic');
    
    // Subtle shadow for all nodes
    const subtleShadow = defs.append('filter')
      .attr('id', 'subtle-shadow')
      .attr('height', '120%');

    subtleShadow.append('feGaussianBlur')
      .attr('in', 'SourceAlpha')
      .attr('stdDeviation', 2)
      .attr('result', 'blur');

    subtleShadow.append('feOffset')
      .attr('in', 'blur')
      .attr('dx', 0)
      .attr('dy', 1)
      .attr('result', 'offsetBlur');

    subtleShadow.append('feComponentTransfer')
      .append('feFuncA')
      .attr('type', 'linear')
      .attr('slope', 0.3);

    const subtleShadowMerge = subtleShadow.append('feMerge');
    subtleShadowMerge.append('feMergeNode').attr('in', 'offsetBlur');
    subtleShadowMerge.append('feMergeNode').attr('in', 'SourceGraphic');
    
    // Gradient for non-leaf nodes
    const nonLeafGradient = defs.append('linearGradient')
      .attr('id', 'non-leaf-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%');
      
    nonLeafGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#4575B4')
      .attr('stop-opacity', 1);
      
    nonLeafGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#6A9EC0')
      .attr('stop-opacity', 1);
      
    // Gradient for leaf nodes - modified to be more distinct from performance indicators
    const leafGradient = defs.append('linearGradient')
      .attr('id', 'leaf-gradient')
      .attr('x1', '0%')
      .attr('y1', '0%')
      .attr('x2', '0%')
      .attr('y2', '100%');
      
    leafGradient.append('stop')
      .attr('offset', '0%')
      .attr('stop-color', '#A569BD') // Purple shade
      .attr('stop-opacity', 1);
      
    leafGradient.append('stop')
      .attr('offset', '100%')
      .attr('stop-color', '#C86862') // Reddish shade
      .attr('stop-opacity', 1);

    const tree = d3.tree().nodeSize([350, 80]);

    let i = 0;
    let selectedNode = null;
    let pickedNode = null; // Track the currently "picked" node
    let update; // Declare update function at higher scope
    
    // Function to highlight a picked node (without centering)
    function highlightPickedNode(d) {
      // Remove highlight from previous picked node
      if (pickedNode) {
        d3.select(`g.node[data-id="${pickedNode.id}"]`)
          .select('rect')
          .classed('picked-node', false)
          .transition()
          .duration(300)
          .attr('stroke-width', 1)
          .attr('stroke-opacity', 0.5);
      }
      
      // Set new picked node
      pickedNode = d;
      
      // Add highlight to new picked node
      const nodeElement = d3.select(`g.node[data-id="${d.id}"]`);
      nodeElement.select('rect')
        .classed('picked-node', true)
        .transition()
        .duration(300)
        .attr('stroke-width', 3)
        .attr('stroke-opacity', 1)
        .attr('stroke', '#ff9800');
    }

    // Context menu setup
    const contextMenu = document.getElementById('context-menu');
    const detailsOverlay = document.getElementById('details-overlay');
    const detailsClose = document.querySelector('.details-close');

    // Close context menu when clicking outside
    document.addEventListener('click', () => {
      contextMenu.style.display = 'none';
    });

    // Close details overlay with X button
    detailsClose.addEventListener('click', () => {
      detailsOverlay.style.display = 'none';
    });

    // ESC key handling is now managed by the global handler

    // Update context menu items visibility based on node type
    function updateContextMenu(d) {
      const toggleExpandItem = document.getElementById('toggle-expand');
      const showSizeItem = document.getElementById('show-size');
      const showRankingsItem = document.getElementById('show-rankings');
      const showInputsItem = document.getElementById('show-inputs');
      
      const isLeafNode = d.data.size === 1;
      
      // For leaf nodes, only show rankings if available
      if (isLeafNode) {
        toggleExpandItem.style.display = 'none';
        showSizeItem.style.display = 'none';
        showRankingsItem.style.display = d.data.ranking ? 'block' : 'none';
        showInputsItem.style.display = 'none';
      } else {
        // For non-leaf nodes, show all options
        toggleExpandItem.style.display = 'block';
        showSizeItem.style.display = 'block';
        showRankingsItem.style.display = d.data.ranking ? 'block' : 'none';
        showInputsItem.style.display = 'block';
      }
    }

    function getAllLeafInputs(node) {
      const leaves = [];
      function traverse(n) {
        if (n.data.size === 1) {
          leaves.push({
            capability: n.data.capability,
            input: n.data.input
          });
        } else {
          const children = n.children || n._children;
          if (children) {
            children.forEach(traverse);
          }
        }
      }
      traverse(node);
      return leaves;
    }

    function showInputs(d) {
      const detailsTitle = document.getElementById('details-title');
      const detailsTables = document.getElementById('details-tables');
      
      detailsTitle.textContent = 'Leaf Node (Instance) Inputs';
      detailsTables.innerHTML = '';

      const leaves = getAllLeafInputs(d);
      
      const inputsInfo = document.createElement('div');
      inputsInfo.innerHTML = `
        <div style="text-align: center; margin: 0 auto 30px; padding: 20px; background-color: #f8f9fa; border-radius: 12px; max-width: 500px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
          <div style="font-size: 20px; font-weight: 600; color: var(--primary-color); margin-bottom: 12px;">
            ${escapeHtml(d.data.capability)}
          </div>
          <div style="color: #546e7a; font-size: 15px; display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="background-color: var(--secondary-color); color: white; border-radius: 20px; padding: 4px 12px; font-weight: 500;">${leaves.length}</span>
            <span>Total Leaf Nodes (Instances)</span>
          </div>
        </div>
      `;
      detailsTables.appendChild(inputsInfo);

      const inputsTable = document.createElement('div');
      
      if (leaves.length > 0) {
        inputsTable.innerHTML = `
          <div class="table-title">Leaf Node (Instance) Inputs</div>
          <table>
            <thead>
              <tr>
                <th style="width: 60px;">#</th>
                <th>Input</th>
              </tr>
            </thead>
            <tbody>
              ${leaves.map((leaf, index) => `
                <tr>
                  <td style="color: #757575; font-weight: 500;">${index + 1}</td>
                  <td style="font-family: 'Inter', monospace; background-color: ${index % 2 === 0 ? '#f8fafc' : '#ffffff'}; padding: 12px 16px;">
                    ${escapeHtml(leaf.input)}
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        `;
      } else {
        inputsTable.innerHTML = `
          <div style="text-align: center; padding: 20px; color: #757575;">
            No leaf node inputs available.
          </div>
        `;
      }
      
      detailsTables.appendChild(inputsTable);
      detailsOverlay.style.display = 'flex';
    }

    function showSizeDetails(d) {
      const detailsTitle = document.getElementById('details-title');
      const detailsTables = document.getElementById('details-tables');
      
      detailsTitle.textContent = 'Instance Number Details';
      detailsTables.innerHTML = '';

      const sizeInfo = document.createElement('div');
      sizeInfo.innerHTML = `
        <div style="text-align: center; margin: 0 auto 30px; padding: 20px; background-color: #f8f9fa; border-radius: 12px; max-width: 500px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
          <div style="font-size: 20px; font-weight: 600; color: var(--primary-color); margin-bottom: 12px;">
            ${escapeHtml(d.data.capability)}
          </div>
          <div style="color: #546e7a; font-size: 15px; display: flex; align-items: center; justify-content: center; gap: 8px;">
            <span style="background-color: var(--primary-color); color: white; border-radius: 20px; padding: 4px 12px; font-weight: 500;">${d.data.size}</span>
            <span>Total Instances</span>
          </div>
        </div>
      `;
      detailsTables.appendChild(sizeInfo);

      const sizeTable = document.createElement('div');
      sizeTable.innerHTML = `
        <div class="table-title">Child Nodes</div>
        <table>
          <thead>
            <tr>
              <th>Distinction</th>
              <th>Capability</th>
              <th>Instance Number</th>
            </tr>
          </thead>
          <tbody>
            ${d.children ? d.children.map(child => `
              <tr>
                <td>${escapeHtml(child.data.distinction || '')}</td>
                <td>${escapeHtml(child.data.capability)}</td>
                <td><span style="font-weight: 500; color: var(--primary-color);">${child.data.size}</span></td>
              </tr>
            `).join('') : d._children ? d._children.map(child => `
              <tr>
                <td>${escapeHtml(child.data.distinction || '')}</td>
                <td>${escapeHtml(child.data.capability)}</td>
                <td><span style="font-weight: 500; color: var(--primary-color);">${child.data.size}</span></td>
              </tr>
            `).join('') : ''}
          </tbody>
        </table>
      `;
      detailsTables.appendChild(sizeTable);
      detailsOverlay.style.display = 'flex';
    }

    function showRankings(d) {
      const detailsTitle = document.getElementById('details-title');
      const detailsTables = document.getElementById('details-tables');
      
      detailsTitle.textContent = 'Model Ranking';
      detailsTables.innerHTML = '';

      const rankingInfo = document.createElement('div');
      rankingInfo.innerHTML = `
        <div style="text-align: center; margin: 0 auto 30px; padding: 20px; background-color: #f8f9fa; border-radius: 12px; max-width: 500px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
          <div style="font-size: 20px; font-weight: 600; color: var(--primary-color); margin-bottom: 12px;">
            ${escapeHtml(d.data.capability)}
          </div>
          <div style="color: #546e7a; font-size: 15px; display: inline-block; background-color: #e3f2fd; border-radius: 20px; padding: 4px 16px;">
            <strong>Metric:</strong> ${currentMetrics}
          </div>
        </div>
      `;
      detailsTables.appendChild(rankingInfo);

      const rankingTable = document.createElement('div');
      
      if (d.data.ranking && d.data.ranking.length > 0) {
        // Convert to array of objects for easier processing
        const rankings = d.data.ranking.map(([model, score]) => ({ model, score }));
        
        // Sort by score in descending order
        rankings.sort((a, b) => b.score - a.score);
        
        // Find the highest score for highlighting
        const maxScore = Math.max(...rankings.map(r => r.score));
        
        rankingTable.innerHTML = `
          <div class="table-title">Model Performance Ranking</div>
          <table>
            <thead>
              <tr>
                <th style="width: 80px;">Rank</th>
                <th>Model</th>
                <th style="width: 120px;">${currentMetrics}</th>
                <th style="width: 180px;">95% Confidence Interval (CI)</th>
              </tr>
            </thead>
            <tbody>
              ${rankings.map((rank, index) => {
                const isTopRank = rank.score === maxScore;
                const rankStyle = isTopRank ? 'background-color: #e8f5e9; font-weight: 600;' : '';
                const scoreColor = isTopRank ? '#2e7d32' : 'var(--primary-color)';
                
                // Check if CI data exists for this model
                let ciHtml = '<td>-</td>';
                if (d.data.CI && d.data.CI[rank.model]) {
                  const ci = d.data.CI[rank.model];
                  if (Array.isArray(ci) && ci.length === 2) {
                    ciHtml = `<td style="font-weight: 500; color: ${scoreColor};">[${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]</td>`;
                  }
                }
                
                return `
                  <tr style="${rankStyle}">
                    <td>
                      ${index === 0 ? 
                        '<span style="display: inline-block; background-color: #ffd700; color: #333; border-radius: 50%; width: 24px; height: 24px; text-align: center; line-height: 24px; font-weight: bold;">1</span>' : 
                        `<span style="font-weight: 500;">${index + 1}</span>`
                      }
                    </td>
                    <td>${escapeHtml(rank.model)}</td>
                    <td style="font-weight: 500; color: ${scoreColor};">${rank.score.toFixed(3)}</td>
                    ${ciHtml}
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        `;
      } else {
        rankingTable.innerHTML = `
          <div style="text-align: center; padding: 20px; color: #757575;">
            No ranking data available for this node.
          </div>
        `;
      }
      
      detailsTables.appendChild(rankingTable);
      detailsOverlay.style.display = 'flex';
    }

    // Function to find node by path
    function findNodeByPath(path) {
      let node = root;
      for (const index of path) {
        if (node.children) {
          node = node.children[index];
        } else if (node._children) {
          node = node._children[index];
        } else {
          return null;
        }
      }
      return node;
    }

    // Search handler
    function handleSearch(searchText) {
      if (!searchText) {
        searchResults.style.display = 'none';
        return;
      }

      const results = searchEngine.search(searchText);
      if (results.length === 0) {
        searchResults.innerHTML = '<div style="padding: 8px; color: #666;">No results found</div>';
        searchResults.style.display = 'block';
        return;
      }

      searchResults.innerHTML = results.map(result => `
        <div class="search-result" style="
          padding: 8px;
          border-bottom: 1px solid #eee;
          cursor: pointer;
          hover: background-color: #f5f5f5;
        ">
          <div style="font-weight: bold;">${escapeHtml(result.capability)}</div>
          ${result.isLeaf && result.input ? `
            <div style="color: #666; font-size: 0.9em; margin-top: 4px;">
              ${escapeHtml(result.input)}
            </div>
          ` : ''}
          <div style="color: #999; font-size: 0.8em; margin-top: 4px;">
            Depth: ${result.depth} | Relevance Score: ${result.score.toFixed(3)}
          </div>
        </div>
      `).join('');

      searchResults.style.display = 'block';

      // Add click handlers to search results
      const resultElements = searchResults.getElementsByClassName('search-result');
      Array.from(resultElements).forEach((el, index) => {
        el.addEventListener('click', () => {
          const node = findNodeByPath(results[index].path);
          if (node) {
            // First collapse all nodes
            function collapseAll(d) {
              if (d.children) {
                d._children = d.children;
                d.children = null;
                d._children.forEach(collapseAll);
              }
            }
            collapseAll(root);

            // Then expand the path to the target node and the node itself
            let current = node;
            const pathToExpand = [node]; // Include the target node itself
            while (current.parent) {
              pathToExpand.unshift(current.parent);
              current = current.parent;
            }
            
            pathToExpand.forEach(d => {
              if (d._children) {
                d.children = d._children;
                d._children = null;
              }
            });

            update(root);
            
            // Highlight the node found in search
            highlightPickedNode(node);
            
            searchResults.style.display = 'none';
            searchInput.value = '';
          }
        });

        // Hover effect
        el.addEventListener('mouseover', () => {
          el.style.backgroundColor = '#f5f5f5';
        });
        el.addEventListener('mouseout', () => {
          el.style.backgroundColor = 'white';
        });
      });
    }

    // Add search input handler for Enter key
    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        handleSearch(e.target.value);
      } else if (e.key === 'Escape') {
        searchInput.value = '';
        searchResults.style.display = 'none';
      }
    });

    d3.json(jsonFile).then(async data => {
      // Initialize frontend search index
      searchEngine.buildIndex(data);

      root = d3.hierarchy(data, d => d.subtrees);
      root.x0 = height / 2;
      root.y0 = 0;

      function collapse(d) {
        if (d.children) {
          d._children = d.children;
          d._children.forEach(collapse);
          d.children = null;
        }
      }

      // Define the update function first
      update = function(source) {
        const treeData = tree(root);
        const nodes = treeData.descendants();
        const links = treeData.links();

        // Moderate vertical spacing between levels (60% of previous value)
        nodes.forEach(d => {
          d.y = d.depth * 230; // Reduced to 60% of previous value (380 * 0.6 ≈ 230)
        });

        // Ensure minimum horizontal spacing between nodes at the same level
        nodes.forEach((node, index) => {
          if (index > 0) {
            let previousNode = nodes[index - 1];
            // Check if nodes are at the same level and too close
            if (node.depth === previousNode.depth) {
              // Calculate minimum spacing based on node widths
              const minSpacing = (previousNode.nodeWidth || 300) / 2 + (node.nodeWidth || 300) / 2 + 50;
              
              // If nodes are too close, move the current node
              if (Math.abs(node.x - previousNode.x) < minSpacing) {
                node.x = previousNode.x + minSpacing;
              }
            }
          }
        });

        const node = g.selectAll('g.node')
          .data(nodes, d => d.id || (d.id = ++i));

        const nodeEnter = node.enter().append('g')
          .attr('class', 'node')
          .attr('data-id', d => d.id) // Add data-id for easier selection
          .attr('transform', d => `translate(${source.x0},${source.y0})`)
          .attr('opacity', 0)
          .on('click', (event, d) => {
            if (d.children) {
              d._children = d.children;
              d.children = null;
            } else if (d._children) {
              d.children = d._children;
              d._children = null;
            }
            update(d);
            
            // Highlight the clicked node
            highlightPickedNode(d);
          })
          .on('contextmenu', (event, d) => {
            event.preventDefault();
            selectedNode = d;
            
            const x = event.clientX;
            const y = event.clientY;
            
            contextMenu.style.left = `${x}px`;
            contextMenu.style.top = `${y}px`;
            
            updateContextMenu(d);
            contextMenu.style.display = 'block';
          });

        // Base dimensions that will be adjusted based on content - extremely compact
        const baseWidth = 240;
        const minHeight = 80; // Significantly reduced minimum height
        const padding = 10; // Minimal padding around content
        
        // Create temporary div to measure text dimensions - with minimal styling
        const measureDiv = document.createElement('div');
        measureDiv.style.position = 'absolute';
        measureDiv.style.visibility = 'hidden';
        measureDiv.style.width = `${baseWidth - padding}px`; // Account for padding
        measureDiv.style.fontFamily = 'Inter, sans-serif';
        measureDiv.style.fontSize = '13px'; // Smaller font
        measureDiv.style.lineHeight = '1.2'; // Tighter line height
        measureDiv.style.wordWrap = 'break-word';
        document.body.appendChild(measureDiv);
        
        // Function to calculate node dimensions based on content - extremely compact
        function calculateNodeDimensions(d) {
            // Create HTML content similar to what will be displayed but with minimal spacing
            let html = '';
            
            if (d.data.size === 1) {
                // Add model-specific performance if a model is selected and ranking data exists
                let modelPerformanceHtml = '';
                if (currentModel && d.data.ranking) {
                    // Check if ranking is an array or an object
                    if (Array.isArray(d.data.ranking)) {
                        // Find the model in the ranking array
                        const modelRanking = d.data.ranking.find(item => item[0] === currentModel);
                        if (modelRanking) {
                            modelPerformanceHtml = `<div style="margin-top: 6px; font-size: 13px;">Performance indicator</div>`;
                        }
                    } else if (typeof d.data.ranking === 'object' && d.data.ranking[currentModel] !== undefined) {
                        modelPerformanceHtml = `<div style="margin-top: 6px; font-size: 13px;">Performance indicator</div>`;
                    }
                }
                
                html = `<div style="padding: 2px;">
                  <div style="font-weight: bold; margin-bottom: 4px;">Instance Input:</div>
                  <div style="margin-bottom: 4px;">${escapeHtml(d.data.input || '')}</div>
                  ${modelPerformanceHtml}
                </div>`;
            } else {
                const distinctionHtml = d.data.distinction ? 
                    `<div style="margin-bottom: 2px; font-size: 14px;">${escapeHtml(d.data.distinction)}</div>` : '';
                
                // Add model-specific performance if a model is selected and ranking data exists
                let modelPerformanceHtml = '';
                if (currentModel && d.data.ranking) {
                    // Function to get color based on score (0-100 scale)
                    const getScoreColor = (score) => {
                        // Normalize score to 0-100 if needed
                        const normalizedScore = score > 1 ? score : score * 100;
                        
                        if (normalizedScore >= 90) return '#4575B4'; // Blue for excellent
                        if (normalizedScore >= 75) return '#92C5E0'; // Light blue for very good
                        if (normalizedScore >= 60) return '#A569BD'; // Purple for good
                        if (normalizedScore >= 45) return '#D9A441'; // Gold for average
                        if (normalizedScore >= 30) return '#C86862'; // Reddish for below average
                        return '#5A5A5A'; // Dark gray for poor
                    };
                    
                    // Check if ranking is an array or an object
                    if (Array.isArray(d.data.ranking)) {
                        // Find the model in the ranking array
                        const modelRanking = d.data.ranking.find(item => item[0] === currentModel);
                        if (modelRanking) {
                            const score = modelRanking[1];
                            
                            // Add CI information if available
                            let ciHtml = '';
                            if (d.data.CI && d.data.CI[currentModel]) {
                                const ci = d.data.CI[currentModel];
                                if (Array.isArray(ci) && ci.length === 2) {
                                    // Use minimal margins to avoid excess space
                                    ciHtml = `<div style="margin-top: 2px; font-size: 13px;">95% CI: [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]</div>`;
                                }
                            }
                            
                            modelPerformanceHtml = `<div style="margin-top: 6px; font-size: 13px;">${score.toFixed(3)} ${currentMetrics}</div>${ciHtml}`;
                        }
                    } else if (typeof d.data.ranking === 'object' && d.data.ranking[currentModel] !== undefined) {
                        // If ranking is an object with model names as keys
                        const score = d.data.ranking[currentModel];
                        
                        // Add CI information if available
                        let ciHtml = '';
                        if (d.data.CI && d.data.CI[currentModel]) {
                            const ci = d.data.CI[currentModel];
                            if (Array.isArray(ci) && ci.length === 2) {
                                // Use minimal margins to avoid excess space
                                ciHtml = `<div style="margin-top: 2px; font-size: 13px;">95% CI: [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]</div>`;
                            }
                        }
                        
                        modelPerformanceHtml = `<div style="margin-top: 6px; font-size: 13px;">${score.toFixed(3)} ${currentMetrics}</div>${ciHtml}`;
                    }
                }
                
                html = `
                    <div style="padding: 2px;">
                        ${distinctionHtml}
                        <div style="margin-bottom: 2px;">${escapeHtml(d.data.capability || '')}</div>
                        <div>${d.data.size} instances</div>
                        ${modelPerformanceHtml}
                    </div>
                `;
            }
            
            measureDiv.innerHTML = html;
            
            // Get content dimensions with almost no extra space
            const contentWidth = measureDiv.scrollWidth + 4;
            const contentHeight = measureDiv.scrollHeight + 4;
            
            // Calculate final dimensions with minimum sizes and minimal padding
            let width = Math.max(contentWidth + padding, baseWidth);
            let height = Math.max(contentHeight + padding, minHeight);
            
            // If the node has CI data for the current model, ensure it has enough width
            // but keep the height as compact as possible
            if (currentModel && d.data.CI && d.data.CI[currentModel]) {
                // Increase width to accommodate both performance and CI boxes
                width = Math.max(width, 300);
            }
            
            return { width, height };
        }
        
        // Add rounded rectangle background with gradient fill - more compact
        nodeEnter.append('rect')
          .attr('rx', 8) // Smaller corner radius
          .attr('ry', 8)
          .attr('fill', d => (d.data.size === 1 ? 'url(#leaf-gradient)' : 'url(#non-leaf-gradient)'))
          .attr('filter', d => d._children ? 'url(#drop-shadow)' : 'url(#subtle-shadow)')
          .attr('stroke', d => d.data.size === 1 ? '#A569BD' : '#4575B4')
          .attr('stroke-width', 1) // Thinner stroke
          .attr('stroke-opacity', 0.5) // More subtle stroke
          .each(function(d) {
              const dims = calculateNodeDimensions(d);
              d.nodeWidth = dims.width;
              d.nodeHeight = dims.height;
              
              d3.select(this)
                .attr('width', dims.width)
                .attr('height', dims.height)
                .attr('x', -dims.width / 2)
                .attr('y', -dims.height / 2);
          });

        // Add foreign object for HTML content with minimal padding
        const foreignObject = nodeEnter.append('foreignObject')
          .style('overflow', 'hidden')
          .each(function(d) {
              const padding = 4; // Minimal padding inside the node
              const width = d.nodeWidth - padding * 2;
              const height = d.nodeHeight - padding * 2;
              
              d3.select(this)
                .attr('x', -width / 2)
                .attr('y', -height / 2)
                .attr('width', width)
                .attr('height', height);
          });

        // Add content with improved styling and minimal padding
        foreignObject.append('xhtml:div')
          .style('width', '100%')
          .style('height', '100%')
          .style('display', 'flex')
          .style('flex-direction', 'column')
          .style('justify-content', 'center')
          .style('align-items', 'center')
          .style('font-size', '13px') // Smaller font
          .style('font-family', 'Inter, sans-serif')
          .style('text-align', 'center')
          .style('line-height', '1.2') // Tighter line height
          .style('word-wrap', 'break-word')
          .style('color', '#ffffff')
          .html(d => {
            if (d.data.size === 1) {
              // Leaf node with minimal padding
              // Add model-specific performance if a model is selected and ranking data exists
              let modelPerformanceHtml = '';
              if (currentModel && d.data.ranking) {
                // Function to get color based on score (0-100 scale)
                const getScoreColor = (score) => {
                  // Normalize score to 0-100 if needed
                  const normalizedScore = score > 1 ? score : score * 100;

                  if (normalizedScore >= 90) return '#4575B4'; // Blue for excellent
                  if (normalizedScore >= 75) return '#92C5E0'; // Light blue for very good
                  if (normalizedScore >= 60) return '#A569BD'; // Purple for good
                  if (normalizedScore >= 45) return '#D9A441'; // Gold for average
                  if (normalizedScore >= 30) return '#C86862'; // Reddish for below average
                  return '#5A5A5A'; // Dark gray for poor
                };

                // Check if ranking is an array or an object
                if (Array.isArray(d.data.ranking)) {
                  // Find the model in the ranking array
                  const modelRanking = d.data.ranking.find(item => item[0] === currentModel);
                                                                        if (modelRanking) {
                    const score = modelRanking[1];
                    const backgroundColor = getScoreColor(score);
                    
                    // Create the performance metric box
                    let performanceHtml = `<div style="margin-top: 2px; font-size: 13px; font-weight: 600; color: white; background-color: ${backgroundColor}; border-radius: 12px; padding: 2px 8px; display: inline-block;">${score.toFixed(3)} ${currentMetrics}</div>`;
                    
                    // Check if CI data exists for this model and create a separate box
                    let ciHtml = '';
                    let hasCIData = false;
                    if (d.data.CI && d.data.CI[currentModel]) {
                      const ci = d.data.CI[currentModel];
                      if (Array.isArray(ci) && ci.length === 2) {
                        hasCIData = true;
                        // Use the same color for the CI box
                        const ciBackgroundColor = backgroundColor;
                        ciHtml = `<div style="margin-top: 1px; margin-left: 2px; font-size: 13px; font-weight: 600; color: white; background-color: ${ciBackgroundColor}; border-radius: 12px; padding: 2px 8px; display: inline-block;">95% CI: [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]</div>`;
                      }
                    }
                    
                    
                    
                    modelPerformanceHtml = `<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; width: 100%;">${performanceHtml}${ciHtml}</div>`;
                  }
                } else if (typeof d.data.ranking === 'object' && d.data.ranking[currentModel] !== undefined) {
                                                                        // If ranking is an object with model names as keys
                  const score = d.data.ranking[currentModel];
                  const backgroundColor = getScoreColor(score);
                  
                  // Create the performance metric box
                  let performanceHtml = `<div style="margin-top: 2px; font-size: 13px; font-weight: 600; color: white; background-color: ${backgroundColor}; border-radius: 12px; padding: 2px 8px; display: inline-block;">${score.toFixed(3)} ${currentMetrics}</div>`;
                  
                  // Check if CI data exists for this model and create a separate box
                  let ciHtml = '';
                  let hasCIData = false;
                  if (d.data.CI && d.data.CI[currentModel]) {
                    const ci = d.data.CI[currentModel];
                    if (Array.isArray(ci) && ci.length === 2) {
                      hasCIData = true;
                      // Use the same color for the CI box
                      const ciBackgroundColor = backgroundColor;
                      ciHtml = `<div style="margin-top: 1px; margin-left: 2px; font-size: 13px; font-weight: 600; color: white; background-color: ${ciBackgroundColor}; border-radius: 12px; padding: 2px 8px; display: inline-block;">95% CI: [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]</div>`;
                    }
                  }
                  
                  
                  
                  modelPerformanceHtml = `<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; width: 100%;">${performanceHtml}${ciHtml}</div>`;
                }
              }
              
              return `<div style="padding: 2px;">
                <div style="font-weight: bold; margin-bottom: 4px; color: #D9A441;">Instance Input:</div>
                <div style="margin-bottom: 4px;">${escapeHtml(d.data.input)}</div>
                ${modelPerformanceHtml}
              </div>`;
            } else {
              // Non-leaf node with instance count and minimal margins
              const distinctionHtml = d.data.distinction ? 
                `<div style="font-weight: 600; color: #D9A441; margin-bottom: 2px; font-size: 14px;">${escapeHtml(d.data.distinction)}</div>` : '';
              
              // Add model-specific performance if a model is selected and ranking data exists
              let modelPerformanceHtml = '';
              if (currentModel && d.data.ranking) {
                // Function to get color based on score (0-100 scale)
                const getScoreColor = (score) => {
                  // Normalize score to 0-100 if needed
                  const normalizedScore = score > 1 ? score : score * 100;
                  
                  if (normalizedScore >= 90) return '#4575B4'; // Blue for excellent
                  if (normalizedScore >= 75) return '#92C5E0'; // Light blue for very good
                  if (normalizedScore >= 60) return '#A569BD'; // Purple for good
                  if (normalizedScore >= 45) return '#D9A441'; // Gold for average
                  if (normalizedScore >= 30) return '#C86862'; // Reddish for below average
                  return '#5A5A5A'; // Dark gray for poor
                };
                
                // Check if ranking is an array or an object
                if (Array.isArray(d.data.ranking)) {
                  // Find the model in the ranking array
                  const modelRanking = d.data.ranking.find(item => item[0] === currentModel);
                                                                        if (modelRanking) {
                    const score = modelRanking[1];
                    const backgroundColor = getScoreColor(score);
                    
                    // Create the performance metric box
                    let performanceHtml = `<div style="margin-top: 2px; font-size: 13px; font-weight: 600; color: white; background-color: ${backgroundColor}; border-radius: 12px; padding: 2px 8px; display: inline-block;">${score.toFixed(3)} ${currentMetrics}</div>`;
                    
                    // Check if CI data exists for this model and create a separate box
                    let ciHtml = '';
                    let hasCIData = false;
                    if (d.data.CI && d.data.CI[currentModel]) {
                      const ci = d.data.CI[currentModel];
                      if (Array.isArray(ci) && ci.length === 2) {
                        hasCIData = true;
                        // Use the same color for the CI box
                        const ciBackgroundColor = backgroundColor;
                        ciHtml = `<div style="margin-top: 1px; margin-left: 2px; font-size: 13px; font-weight: 600; color: white; background-color: ${ciBackgroundColor}; border-radius: 12px; padding: 2px 8px; display: inline-block;">95% CI: [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]</div>`;
                      }
                    }
                    
                    
                    
                    modelPerformanceHtml = `<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; width: 100%;">${performanceHtml}${ciHtml}</div>`;
                  }
                } else if (typeof d.data.ranking === 'object' && d.data.ranking[currentModel] !== undefined) {
                                                                        // If ranking is an object with model names as keys
                  const score = d.data.ranking[currentModel];
                  const backgroundColor = getScoreColor(score);
                  
                  // Create the performance metric box
                  let performanceHtml = `<div style="margin-top: 2px; font-size: 13px; font-weight: 600; color: white; background-color: ${backgroundColor}; border-radius: 12px; padding: 2px 8px; display: inline-block;">${score.toFixed(3)} ${currentMetrics}</div>`;
                  
                  // Check if CI data exists for this model and create a separate box
                  let ciHtml = '';
                  let hasCIData = false;
                  if (d.data.CI && d.data.CI[currentModel]) {
                    const ci = d.data.CI[currentModel];
                    if (Array.isArray(ci) && ci.length === 2) {
                      hasCIData = true;
                      // Use the same color for the CI box
                      const ciBackgroundColor = backgroundColor;
                      ciHtml = `<div style="margin-top: 1px; margin-left: 2px; font-size: 13px; font-weight: 600; color: white; background-color: ${ciBackgroundColor}; border-radius: 12px; padding: 2px 8px; display: inline-block;">95% CI: [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]</div>`;
                    }
                  }
                  
                  
                  
                  modelPerformanceHtml = `<div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; width: 100%;">${performanceHtml}${ciHtml}</div>`;
                }
              }
              
              return `
                <div style="padding: 2px;">
                  ${distinctionHtml}
                  <div style="font-weight: 500; margin-bottom: 2px;">${escapeHtml(d.data.capability)}</div>
                  <div style="font-size: 12px; font-weight: 600; background-color: rgba(255, 255, 255, 0.25); border-radius: 10px; padding: 1px 6px; display: inline-block;">${d.data.size} instances</div>
                  ${modelPerformanceHtml}
                </div>
              `;
            }
          });
          
        // Remove the temporary measurement div
        document.body.removeChild(measureDiv);

        const nodeUpdate = nodeEnter.merge(node);
        
        // Smooth transition for node position and appearance
        nodeUpdate.transition()
          .duration(750)
          .attr('transform', d => `translate(${d.x},${d.y})`)
          .attr('opacity', 1);

        // Update rectangle appearance
        nodeUpdate.select('rect')
          .attr('fill', d => (d.data.size === 1 ? 'url(#leaf-gradient)' : 'url(#non-leaf-gradient)'))
          .attr('filter', d => d._children ? 'url(#drop-shadow)' : 'url(#subtle-shadow)')
          .attr('stroke', d => d.data.size === 1 ? '#A569BD' : '#4575B4');

        // Smooth exit transition with fade out
        const nodeExit = node.exit()
          .transition()
          .duration(600)
          .attr('transform', d => `translate(${source.x},${source.y})`)
          .attr('opacity', 0)
          .remove();

        const link = g.selectAll('path.link')
          .data(links, d => d.target.id);

        const linkEnter = link.enter().insert('path', 'g')
          .attr('class', 'link')
          .attr('d', d => {
            const o = { x: source.x0, y: source.y0 };
            return diagonal(o, o);
          })
          .attr('fill', 'none')
          .attr('stroke', '#92C5E0')
          .attr('stroke-width', 2)
          .attr('stroke-opacity', 0.7)
          .attr('stroke-linecap', 'round')
          .attr('stroke-linejoin', 'round')
          .attr('opacity', 0);

        const linkUpdate = linkEnter.merge(link);
        linkUpdate.transition()
          .duration(750)
          .attr('d', d => diagonal(d.source, d.target))
          .attr('opacity', 1);

        const linkExit = link.exit()
          .transition()
          .duration(600)
          .attr('d', d => {
            const o = { x: source.x, y: source.y };
            return diagonal(o, o);
          })
          .attr('opacity', 0)
          .remove();

        nodes.forEach(d => {
          d.x0 = d.x;
          d.y0 = d.y;
        });
      }

      function diagonal(s, d) {
        return `M ${s.x} ${s.y}
                C ${(s.x + d.x) / 2} ${s.y},
                  ${(s.x + d.x) / 2} ${d.y},
                  ${d.x} ${d.y}`;
      }

      // Set up context menu handlers
      document.getElementById('toggle-expand').addEventListener('click', () => {
        if (selectedNode) {
          if (selectedNode.children) {
            selectedNode._children = selectedNode.children;
            selectedNode.children = null;
          } else if (selectedNode._children) {
            selectedNode.children = selectedNode._children;
            selectedNode._children = null;
          }
          update(selectedNode);
          
          // Highlight the toggled node
          highlightPickedNode(selectedNode);
        }
      });

      document.getElementById('show-size').addEventListener('click', () => {
        if (selectedNode && selectedNode.data.size > 1) {
          showSizeDetails(selectedNode);
        }
      });

      document.getElementById('show-rankings').addEventListener('click', () => {
        if (selectedNode && selectedNode.data.ranking) {
          showRankings(selectedNode);
        }
      });

      document.getElementById('show-inputs').addEventListener('click', () => {
        if (selectedNode && selectedNode.data.size > 1) {
          showInputs(selectedNode);
        }
      });

      // Now that update is defined, initialize the tree
      if (root.children) {
        root.children.forEach(collapse);
      }

      update(root);
      
      // Just highlight the root node
      highlightPickedNode(root);
      
      // Always initialize the zoom behavior with the correct transform
      // Using the same constants for consistent positioning
      svg.call(
        d3.zoom().transform,
        d3.zoomIdentity.translate(INITIAL_X, INITIAL_Y)
      );
      
      // If we're switching benchmarks/models, reset the view position
      // This ensures the tree is always visible at the initial position after switching
      if (window.resetViewPositionAfterLoad) {
        // Reset the flag
        window.resetViewPositionAfterLoad = false;
        
        // Reset the zoom transform to show the root node in the initial position
        svg.transition()
          .duration(750) // Smooth transition over 750ms
          .call(
            d3.zoom().transform,
            d3.zoomIdentity.translate(INITIAL_X, INITIAL_Y).scale(1) // Reset scale to 1 as well
          );
      }

      // Simulate clicking the root node
      if (root._children) {
        root.children = root._children;
        root._children = null;
        update(root);
      }
    }).catch(error => {
      if (error.name === 'AbortError') {
        console.log('Loading aborted due to dataset switch');
      } else {
        console.error('Error loading data:', error);
      }
    }).finally(() => {
      // Re-enable all buttons and clear loading state
      document.querySelectorAll('.button').forEach(btn => {
        btn.disabled = false;
      });
      isLoading = false;
      currentLoadingAbortController = null;
    });
  }
});