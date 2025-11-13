"""
Test script for the stock identifier tool.
This demonstrates how to use the identify_stocks_in_conversation tool.
"""

from tools.stock_identifier_tool import identify_stocks_in_conversation

# Example conversation texts
example_conversations = [
    # Real transcription from diarization output
    """speaker_0:
speaker_0:喂啊鄭女時點啊。😊
speaker_1:哇跌咁多啊樓生今日啊。😊
speaker_0:個別啦，又唔繫(係)好跌好多啫。😮
speaker_0:哦，例如咧。
speaker_1:咁。😊
speaker_0:你唔冇聽人哋講就係鐵斯得噶。😊
speaker_1:啊嗰個電子仔可唔可以買啊。😊
speaker_0:電子仔啊電子仔升得多嘛，誒二百五啊嘛，咁，你又破唔到頂，所以咪唔好買咯。😊
speaker_0:啊哦，咁，其實有啲都要走(賣)嘅之前，你又唔肯走(賣)啫，你貪心啫。😊
speaker_1:咁。😮
speaker_0:好貪心快手你擔心一滯啊，有錢賺你都唔肯走(賣)。😊
speaker_1:哇阿爸爸(阿里巴巴)就慘到咁多啊。😊
speaker_0:你爸爸(阿里巴巴)有錢賺你又系唔肯走(賣)。
speaker_1:冇啊，第二朝都已經跌落嚟噶啦，買就。😊
speaker_0:嗰晚又一站咧，即刻。😮
speaker_0:誒第二第二日不過你媽爸(阿里巴巴)系買得貴啲嘅相對。
speaker_1:咁而家可唔可以追佢啊。
speaker_0:得唔得啊，你等一等先啦，應該會反彈嘅，八百到，你試下睇下一百五十。😊
speaker_0:一個唔系七蚊啦嚇。
speaker_1:啊唔系新而五啊。😊
speaker_0:都繫(係)系啦。
speaker_0:系咩我唔知睇睇先嚇，我冇睇噶，系啊，系阿巴巴(阿里巴巴)會金啲，咁啊咁唔好搞住啊巴巴(阿里巴巴)都要等等啦，咁啊，佢哋一百五一百五十蚊啦。😊
speaker_1:哦，咁啊快走(賣)咧，快走(賣)。😊
speaker_0:快手唔好搞住咯，快手快手高位(價位)出咗啊，你冇走(賣)啊，大傢伙啊。😊
speaker_1:系咯，系咯。
speaker_0:系啊。
speaker_1:咁啊。
speaker_0:出咗咪出咗啊，即系其實呢個。😊
speaker_0:真系就一陣啫嚇估計都要升嘅快手你要等翻七十。😊
speaker_0:
speaker_0:誒七系九蚊反彈啦搏系。
speaker_1:哦。😮
speaker_1:咁咪買只(隻股票)雙通啲人個個都睇住相通(商湯)噶，點解唔得意噶。
speaker_0:唔系雙湯(商湯)咧嚇。
speaker_1:佢話，佢佢佢都系科技股，但系佢。
speaker_0:系。😊
speaker_1:誒。
speaker_0:做乜啊。
speaker_1:
speaker_1:啊，我又去咁。
speaker_0:你咁啦。😊
speaker_0:你即偷翻呢個包包同埋呢個。
speaker_1:翻就。
speaker_0:誒快手啦嚇， o k 哦。😊
speaker_1:好唔該曬，好好，拜拜。😊
"""
]

def test_stock_identifier():
    """Test the stock identifier tool with example conversations."""
    
    print("="*80)
    print("Testing Stock Identifier Tool")
    print("="*80)
    
    for i, conversation in enumerate(example_conversations, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}:")
        print(f"{'='*80}")
        print(f"\nConversation:")
        print(conversation.strip())
        print(f"\n{'-'*80}")
        print("Analysis Result:")
        print(f"{'-'*80}")
        
        # Call the tool (it's a StructuredTool, so use .invoke())
        result = identify_stocks_in_conversation.invoke({"conversation_text": conversation})
        print(result)
        print()

if __name__ == "__main__":
    test_stock_identifier()

