// Global variables
let conversations = [];
let currentConversationIndex = 0;
let currentLanguage = '';

// DOM Elements
const languageSelect = document.getElementById('language-select');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const conversationCounter = document.getElementById('conversation-counter');
const sceneDescription = document.getElementById('scene-description');
const conversationContent = document.getElementById('conversation-content');
const metadataContent = document.getElementById('metadata-content');
const fileLinksContent = document.getElementById('file-links-content');

// Available languages
const languages = [
    { code: 'af-ZA', name: 'Afrikaans (South Africa)' },
    { code: 'ar-SA', name: 'Arabic (Saudi Arabia)' },
    { code: 'ca-ES', name: 'Catalan (Spain)' },
    { code: 'cs-CZ', name: 'Czech (Czech Republic)' },
    { code: 'cy-GB', name: 'Welsh (United Kingdom)' },
    { code: 'da-DK', name: 'Danish (Denmark)' },
    { code: 'de-AT', name: 'German (Austria)' },
    { code: 'de-DE', name: 'German (Germany)' },
    { code: 'el-GR', name: 'Greek (Greece)' },
    { code: 'en-AU', name: 'English (Australia)' },
    { code: 'en-CA', name: 'English (Canada)' },
    { code: 'en-GB', name: 'English (United Kingdom)' },
    { code: 'en-NZ', name: 'English (New Zealand)' },
    { code: 'en-US', name: 'English (United States)' },
    { code: 'en-ZA', name: 'English (South Africa)' },
    { code: 'es-ES', name: 'Spanish (Spain)' },
    { code: 'es-US', name: 'Spanish (United States)' },
    { code: 'eu-ES', name: 'Basque (Spain)' },
    { code: 'fi-FI', name: 'Finnish (Finland)' },
    { code: 'fo-FO', name: 'Faroese (Faroe Islands)' },
    { code: 'fr-CA', name: 'French (Canada)' },
    { code: 'fr-FR', name: 'French (France)' },
    { code: 'he-IL', name: 'Hebrew (Israel)' },
    { code: 'hr-HR', name: 'Croatian (Croatia)' },
    { code: 'it-IT', name: 'Italian (Italy)' },
    { code: 'ja-JP', name: 'Japanese (Japan)' },
    { code: 'ko-KR', name: 'Korean (South Korea)' },
    { code: 'nb-NO', name: 'Norwegian Bokmål (Norway)' },
    { code: 'nl-BE', name: 'Dutch (Belgium)' },
    { code: 'nl-NL', name: 'Dutch (Netherlands)' },
    { code: 'pl-PL', name: 'Polish (Poland)' },
    { code: 'pt-BR', name: 'Portuguese (Brazil)' },
    { code: 'pt-PT', name: 'Portuguese (Portugal)' },
    { code: 'ru-RU', name: 'Russian (Russia)' },
    { code: 'sk-SK', name: 'Slovak (Slovakia)' },
    { code: 'sl-SI', name: 'Slovenian (Slovenia)' },
    { code: 'sv-SE', name: 'Swedish (Sweden)' },
    { code: 'uk-UA', name: 'Ukrainian (Ukraine)' },
    { code: 'zh-CN', name: 'Chinese (China)' }
];

// Initialize the application
function init() {
    populateLanguageSelector();
    setupEventListeners();
}

// Populate the language selector dropdown
function populateLanguageSelector() {
    languages.forEach(language => {
        const option = document.createElement('option');
        option.value = language.code;
        option.textContent = language.name;
        languageSelect.appendChild(option);
    });
}

// Set up event listeners
function setupEventListeners() {
    languageSelect.addEventListener('change', handleLanguageChange);
    prevBtn.addEventListener('click', showPreviousConversation);
    nextBtn.addEventListener('click', showNextConversation);
}

// Handle language change
async function handleLanguageChange() {
    currentLanguage = languageSelect.value;
    
    if (!currentLanguage) {
        resetView();
        return;
    }
    
    try {
        // Show loading state
        conversationContent.innerHTML = '<p>Loading conversations...</p>';
        
        // Fetch conversations for the selected language
        conversations = await fetchConversations(currentLanguage);
        
        if (conversations.length > 0) {
            currentConversationIndex = 0;
            updateNavigationButtons();
            displayConversation(conversations[currentConversationIndex]);
            updateFileLinks(currentLanguage);
        } else {
            conversationContent.innerHTML = '<p>No conversations found for this language</p>';
            resetNavigationButtons();
        }
    } catch (error) {
        console.error('Error loading conversations:', error);
        conversationContent.innerHTML = `<p>Error loading conversations: ${error.message}</p>`;
        resetNavigationButtons();
    }
}

// Fetch conversations for a specific language
async function fetchConversations(languageCode) {
    try {
        const response = await fetch(`../batch_files/${languageCode}/augmented_batch_conversations_transformed.jsonl`);
        const text = await response.text();
        
        // Parse JSONL (each line is a separate JSON object)
        return text.split('\n')
            .filter(line => line.trim() !== '')
            .map(line => JSON.parse(line));
    } catch (error) {
        console.error('Error fetching conversations:', error);
        throw new Error(`Could not load conversations for ${languageCode}`);
    }
}

// Display a conversation
function displayConversation(conversation) {
    // Update scene description
    sceneDescription.textContent = conversation.scene || 'No scene description available';
    
    // Update conversation content
    conversationContent.innerHTML = '';
    
    if (conversation.conversation && conversation.conversation.length > 0) {
        conversation.conversation.forEach(turn => {
            const turnElement = document.createElement('div');
            turnElement.className = 'conversation-turn';
            
            const speakerElement = document.createElement('div');
            speakerElement.className = 'speaker';
            if (turn.is_aac_user) {
                speakerElement.classList.add('aac-user');
                speakerElement.textContent = `${turn.speaker} (AAC User)`;
            } else {
                speakerElement.textContent = turn.speaker;
            }
            
            turnElement.appendChild(speakerElement);
            
            if (turn.is_aac_user && turn.utterance_intended) {
                const intendedElement = document.createElement('div');
                intendedElement.className = 'utterance intended';
                intendedElement.textContent = `Intended: ${turn.utterance_intended}`;
                turnElement.appendChild(intendedElement);
                
                const actualElement = document.createElement('div');
                actualElement.className = 'utterance';
                actualElement.textContent = `Actual: ${turn.utterance}`;
                turnElement.appendChild(actualElement);
            } else {
                const utteranceElement = document.createElement('div');
                utteranceElement.className = 'utterance';
                utteranceElement.textContent = turn.utterance;
                turnElement.appendChild(utteranceElement);
            }
            
            conversationContent.appendChild(turnElement);
        });
    } else {
        conversationContent.innerHTML = '<p>No conversation data available</p>';
    }
    
    // Update metadata
    updateMetadata(conversation);
    
    // Update conversation counter
    conversationCounter.textContent = `Conversation ${currentConversationIndex + 1} of ${conversations.length}`;
}

// Update metadata display
function updateMetadata(conversation) {
    metadataContent.innerHTML = '';
    
    // Display template ID if available
    if (conversation.template_id !== undefined) {
        const templateElement = document.createElement('p');
        templateElement.innerHTML = `<strong>Template ID:</strong> ${conversation.template_id}`;
        metadataContent.appendChild(templateElement);
    }
    
    // Display any other metadata
    if (conversation.metadata) {
        Object.entries(conversation.metadata).forEach(([key, value]) => {
            const metadataElement = document.createElement('p');
            metadataElement.innerHTML = `<strong>${key}:</strong> ${value}`;
            metadataContent.appendChild(metadataElement);
        });
    }
    
    // If no metadata is available
    if (metadataContent.children.length === 0) {
        metadataContent.innerHTML = '<p>No metadata available</p>';
    }
}

// Update file links
function updateFileLinks(languageCode) {
    fileLinksContent.innerHTML = '';
    
    const files = [
        { name: 'Augmented Conversations', path: `../batch_files/${languageCode}/augmented_batch_conversations_transformed.jsonl` },
        { name: 'Batch Output', path: `../batch_files/${languageCode}/` }
    ];
    
    files.forEach(file => {
        const linkElement = document.createElement('a');
        linkElement.href = file.path;
        linkElement.textContent = file.name;
        linkElement.target = '_blank';
        fileLinksContent.appendChild(linkElement);
    });
}

// Show previous conversation
function showPreviousConversation() {
    if (currentConversationIndex > 0) {
        currentConversationIndex--;
        displayConversation(conversations[currentConversationIndex]);
        updateNavigationButtons();
    }
}

// Show next conversation
function showNextConversation() {
    if (currentConversationIndex < conversations.length - 1) {
        currentConversationIndex++;
        displayConversation(conversations[currentConversationIndex]);
        updateNavigationButtons();
    }
}

// Update navigation buttons state
function updateNavigationButtons() {
    prevBtn.disabled = currentConversationIndex === 0;
    nextBtn.disabled = currentConversationIndex === conversations.length - 1;
}

// Reset navigation buttons
function resetNavigationButtons() {
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    conversationCounter.textContent = 'Conversation 0 of 0';
}

// Reset the view
function resetView() {
    sceneDescription.textContent = 'No conversation selected';
    conversationContent.innerHTML = '<p>Select a language and conversation to view</p>';
    metadataContent.innerHTML = '<p>Select a language to view conversations</p>';
    fileLinksContent.innerHTML = '';
    resetNavigationButtons();
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);
